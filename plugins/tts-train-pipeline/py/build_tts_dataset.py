#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tts_dataset.py —— 把「台词清单 + 语音文件」变成 GPT-SoVITS 能直接吃的数据集。

与 dsh-train-pipeline 的 build_dataset.py 的**关键差异**（不是偏好，是硬约束）：

    DDSP/SVC 只要音频，所以超长可以按能量谷切开；
    GPT-SoVITS 要「音频 ↔ 台词」严格对齐，**一旦切分文本就对不上了**。
    因此本脚本**绝不切分**，超长的整条丢弃并计数。这是 TTS 与 SVC 最本质的分野。

两种输入模式：
    1) 台词驱动（galgame 旗舰路线）
       --lines lines.jsonl --voice-dir X:/unpack/voice --speaker 芳乃
       按 speaker 字段精确挑句子，voice 字段模糊匹配音频文件。
       ★ 这是 DDSP 那条链路上被白白丢掉的资产：台词本来就在手里。

    2) 目录驱动（语音按角色分好目录，但没台词）
       --voice-only --voice-dir X:/unpack/voice
       一级子目录即角色（「０１・芳乃」→「芳乃」），无文本。
       这种语料本脚本只规格化，文本留给 asr_label.py 回标。

输出：
    <out>/wav/*.wav      32kHz / 单声道 / PCM16
    <out>/list.txt       wav名|说话人|语言|文本      ← GPT-SoVITS 的标准输入格式
    <out>/build.json     统计与逐条溯源

依赖：**只依赖 ffmpeg 与 Python 标准库**（wave/json/re）。刻意不用 numpy/soundfile，
      因为本插件可能被要求在 GPT-SoVITS 的 venv、系统 Python、甚至纯净 Python 里跑。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import wave

AUDIO_EXT = (".wav", ".ogg", ".mp3", ".flac", ".opus", ".m4a", ".aac", ".wma")

# list.txt 用 | 分隔且解包成 4 段，文本里出现 | 会让 1-get-text.py 抛异常后跳过整句
BAR_RE = re.compile(r"[|｜]")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WS_RE = re.compile(r"[ \t\u3000]+")

KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
LATIN_RE = re.compile(r"[A-Za-z]")

# 清洗后「没有任何可读字符」的判定：只剩标点/空白/符号就该丢
MEANINGFUL_RE = re.compile(r"[\w\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", re.UNICODE)


# ── 文本 ──────────────────────────────────────────────────────────────────

def clean_text(s: str) -> str:
    """把台词洗成能安全塞进 list.txt 的一行。"""
    if not s:
        return ""
    t = str(s)
    t = t.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    t = CTRL_RE.sub("", t)
    t = BAR_RE.sub("，", t)          # 竖线是分隔符，替换掉而不是删掉，保留断句感
    t = WS_RE.sub(" ", t).strip()
    t = t.strip("　 ")
    return t


def is_meaningful(t: str) -> bool:
    return bool(t) and bool(MEANINGFUL_RE.search(t))


def guess_lang(text: str, fallback: str) -> str:
    """按字符构成猜语言。假名优先（日文里汉字和假名混排，必须先判假名）。"""
    if KANA_RE.search(text):
        return "ja"
    if HANGUL_RE.search(text):
        return "ko"
    if HAN_RE.search(text):
        return "zh"
    if LATIN_RE.search(text):
        return "en"
    return fallback


# ── 名称 ──────────────────────────────────────────────────────────────────

def safe_name(s: str, limit: int = 70) -> str:
    s = re.sub(r'[\\/:*?"<>|]', "_", str(s)).strip()
    s = s.strip(". ")
    return s[:limit] or "unnamed"


def strip_index_prefix(s: str) -> str:
    """「０１・芳乃」→「芳乃」;「01_fangnai」→「fangnai」。前缀是序号，不是名字。"""
    t = re.sub(r"^[０-９0-9]{1,3}\s*[・･_.\-、，,．]\s*", "", s).strip()
    return t or s


def audio_files_under(d: str):
    out = []
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.lower().endswith(AUDIO_EXT):
                out.append(os.path.join(root, f))
    return sorted(out)


def build_index(voice_dir: str):
    """文件名 → 完整路径，键做小写去扩展名处理（沿用已验证的模糊匹配策略）。"""
    idx = {}
    for root, _dirs, files in os.walk(voice_dir):
        base = os.path.basename(root).lower()
        for f in files:
            if not f.lower().endswith(AUDIO_EXT):
                continue
            stem = re.sub(r"\.[^.]+$", "", f).lower()
            full = os.path.join(root, f)
            idx.setdefault(stem, full)
            idx.setdefault(f.lower(), full)
            idx.setdefault(base + "/" + stem, full)
    return idx


def read_lines(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ── 音频（ffmpeg + wave，零第三方依赖） ───────────────────────────────────

def wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return 0.0


def end_loudness(path: str, win_sec: float = 0.04):
    """
    收尾处电平：最后一个有声采样前 40ms 的 RMS，相对句内 90 分位（dB）。

    为什么需要它：尾 50ms 电平在**推理产物**上不可用（推理端固定补 300ms 纯零，
    恒为 -240dBFS）；但在**训练集**上它是最好的「句尾有没有被切断」判据。
    真人自然收尾约 -24dB；>-12dB 说明句子停在满电平，就是切尾。
    实测：ogg 原声 -24.0dB（>-12dB 占 7%）｜旧芳乃集 -9.5dB（占 64%）｜留尾重建 -24.5dB（占 4%）。
    """
    import array as _array
    import math as _math
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate() or 1
            raw = w.readframes(w.getnframes())
    except Exception:
        return None
    if not raw:
        return None
    a = _array.array("h")
    try:
        a.frombytes(raw)
    except Exception:
        return None
    if not len(a):
        return None
    a = [abs(x) for x in a]
    peak = max(a)
    if peak <= 0:
        return None
    thr = peak * 0.01
    last = len(a) - 1
    while last > 0 and a[last] < thr:
        last -= 1
    k = max(1, int(win_sec * sr))
    seg = a[max(0, last - k + 1): last + 1]
    rms = _math.sqrt(sum(x * x for x in seg) / float(len(seg))) + 1e-9
    srt = sorted(a)
    ref = srt[int(0.9 * (len(srt) - 1))] + 1e-9
    return 20 * _math.log10(rms / ref)


def ffmpeg_convert(ffmpeg: str, src: str, dst: str, sr: int, trim: bool,
                   tail_pad: float = 0.0, trim_tail: bool = False, timeout: int = 180):
    """
    转成 sr/单声道/PCM16。默认**只裁头部静音，保留句尾自然衰减**，末尾兜一段静音底。

    ★ 尾部静音不能裁（2026-10-24 实测，教科书级坑）：
      `areverse + silenceremove(start_periods=1)` 并不是「切在阈值上」，而是把尾部
      250~400ms 的真实语音一起削掉。实测 `yos001_001.ogg`：
        原声 3.04s / 尾静音 82ms / 收尾电平 -42.6dB
        → 裁后 2.64s / 尾静音 0ms / 收尾电平 **+0.5dB**
      即每条都停在满电平。用这种数据训出来的模型会「切句尾」（末字被吞、尾音没有衰减）。
      芳乃线的 383 条就是这样：100% 尾部静音 = 0ms，模型末三字完整念出率只有 33%。
      正确处理：只裁头 + `apad` 兜静音底 + 保留原衰减。`trim_tail=True` 仅用于复现旧数据集。
    """
    filters = []
    if trim:
        # 头部门槛 -50dB 比原来的 -45dB 更保守，避免削掉起音的弱辅音
        filters.append("silenceremove=start_periods=1:start_duration=0.10:start_threshold=-50dB")
    if trim and trim_tail:
        filters += ["areverse",
                    "silenceremove=start_periods=1:start_duration=0.10:start_threshold=-45dB",
                    "areverse"]
    if tail_pad and tail_pad > 0:
        filters.append("apad=pad_dur=%.3f" % tail_pad)
    args = [ffmpeg, "-y", "-v", "error", "-nostdin", "-i", src,
            "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le"]
    if filters:
        args += ["-af", ",".join(filters)]
    args += [dst]
    try:
        p = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg 超时"
    if p.returncode != 0:
        msg = (p.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        return False, (msg[-1] if msg else f"ffmpeg exit {p.returncode}")
    if not os.path.isfile(dst) or os.path.getsize(dst) < 1000:
        return False, "ffmpeg 产出为空（源文件可能整段静音或已损坏）"
    return True, ""


def ffprobe_for(ffmpeg: str) -> str:
    """从 ffmpeg 路径推 ffprobe；推不出来就用 PATH 里的。"""
    d = os.path.dirname(ffmpeg) if ffmpeg else ""
    for name in ("ffprobe.exe", "ffprobe"):
        p = os.path.join(d, name) if d else name
        if os.path.isfile(p):
            return p
    return "ffprobe"


def source_duration(ffmpeg: str, src: str) -> float:
    """源文件时长。只在 ffmpeg 产出异常时调用（失败分支才付这个开销）。"""
    try:
        p = subprocess.run(
            [ffprobe_for(ffmpeg), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", src],
            capture_output=True, timeout=60)
        return float((p.stdout or b"").decode().strip() or 0)
    except Exception:
        return 0.0


def copy_or_convert(ffmpeg: str, src: str, dst: str, sr: int, trim: bool, min_sec: float,
                    max_sec: float, tail_pad: float = 0.0, trim_tail: bool = False):
    """
    返回 (ok, seconds, reason)。
    ★ 不切分：超长的直接判死，理由写清楚 —— 切了文本就错位了。
    """
    ok, err = ffmpeg_convert(ffmpeg, src, dst, sr, trim, tail_pad=tail_pad, trim_tail=trim_tail)
    if not ok:
        # 「产出为空」多半不是坏文件，而是源本身极短、裁完静音就没了。
        # 探一下源时长，短的归到 skipped_short，别混进 skipped_error 误导排查。
        if "产出为空" in err:
            sd = source_duration(ffmpeg, src)
            if sd and sd < max(min_sec, 1.0):
                return False, sd, f"短于下限 {min_sec}s（源仅 {sd:.2f}s，裁静音后为空）"
        return False, 0.0, err
    dur = wav_duration(dst)
    if dur <= 0:
        return False, 0.0, "无法读取时长"
    if dur < min_sec:
        try:
            os.remove(dst)
        except OSError:
            pass
        return False, dur, f"短于下限 {min_sec}s（{dur:.2f}s）"
    if dur > max_sec:
        try:
            os.remove(dst)
        except OSError:
            pass
        return False, dur, f"超长 {max_sec}s（{dur:.2f}s）—— TTS 不切分，切了文不对音"
    return True, dur, ""


# ── 主流程 ────────────────────────────────────────────────────────────────

def plan_jobs(args):
    """
    返回 [(speaker, [(src, text, scene)])]。
    text 为空串 = 无台词，只规格化（供后续 ASR 回标）。
    """
    if args.voice_only:
        subs = sorted(
            d for d in (os.path.join(args.voice_dir, x) for x in os.listdir(args.voice_dir))
            if os.path.isdir(d)
        ) if os.path.isdir(args.voice_dir) else []
        want = [s.strip() for s in args.speaker.split(",") if s.strip()] if args.speaker else []

        if subs:
            jobs = []
            for d in subs:
                name = strip_index_prefix(os.path.basename(d))
                if want and not any(w == name or w in os.path.basename(d) for w in want):
                    continue
                files = audio_files_under(d)
                if files:
                    jobs.append((name, [(f, "", "") for f in files]))
            return jobs

        # 扁平目录：全归同一个角色
        if not args.speaker:
            return []
        files = audio_files_under(args.voice_dir)
        return [(args.speaker.strip(), [(f, "", "") for f in files])] if files else []

    # 台词驱动
    rows = read_lines(args.lines)
    if not rows:
        return []
    idx = build_index(args.voice_dir) if args.voice_dir else {}
    want = [s.strip() for s in args.speaker.split(",") if s.strip()] if args.speaker else []

    buckets = {}
    for r in rows:
        sp = str(r.get("speaker") or r.get("name") or "").strip()
        if not sp:
            sp = ""          # 旁白：保留，便于统计「有多少台词没有语音」
        if want and "all" not in [w.lower() for w in want] and sp not in want:
            continue
        # ★ 优先用精确路径（extract_scn_text.py 抽出来的 voice_path），
        #   比按名字模糊匹配可靠得多：剧本里的 voice ID 与文件名的对应关系是引擎级的
        src = str(r.get("voice_path") or "")
        if not src or not os.path.isfile(src):
            src = ""
            v = r.get("voice") or r.get("vofile") or r.get("voice_id")
            if isinstance(v, str) and v:
                key = re.sub(r"\.[^.]+$", "", v).lower()
                src = idx.get(key) or idx.get(os.path.basename(v).lower()) or ""
                if not src and os.path.isfile(v):
                    src = v
        text = clean_text(r.get("text") or r.get("message") or "")
        # 没有音频的行也放进来：让「台词有、语音缺」能被统计到，而不是无声消失
        buckets.setdefault(sp, []).append((src, text, str(r.get("scene") or "")))

    jobs = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="构造 GPT-SoVITS 数据集（保留台词）")
    ap.add_argument("--lines", default="", help="lines.jsonl（台词驱动模式）")
    ap.add_argument("--voice-dir", default="", help="语音文件目录")
    ap.add_argument("--voice-only", action="store_true", help="目录驱动：一级子目录=角色，无文本")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--speaker", default="", help="角色名，逗号分隔；all=全部有语音的角色")
    ap.add_argument("--lang", default="zh", help="zh/ja/en/ko/yue；--lang-auto 可覆盖")
    ap.add_argument("--lang-auto", action="store_true", help="按文本字符构成自动判语言")
    ap.add_argument("--sample-rate", type=int, default=32000)
    ap.add_argument("--min-sec", type=float, default=2.0,
                    help="可训时长下限。galgame 单句多在 1.5~4s，默认 2.0 而非官方建议的 3.0")
    ap.add_argument("--max-sec", type=float, default=10.0)
    ap.add_argument("--min-chars", type=int, default=2, help="台词最少字数，太短训不出")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--limit", type=int, default=0, help="每个角色最多出多少条（调试用）")
    ap.add_argument("--no-trim", action="store_true", help="不裁任何静音（头部也不裁）")
    ap.add_argument("--tail-pad", type=float, default=0.16,
                    help="末尾兜底静音秒数。训练集必须留尾，默认 0.16s；设 0 关闭")
    ap.add_argument("--trim-tail", action="store_true",
                    help="★仅复现旧数据集用：裁尾部静音。实测会把 250~400ms 真实语音一起削掉，"
                         "训出来的模型会「切句尾」，不要在生产数据集上打开")
    ap.add_argument("--allow-empty-text", action="store_true",
                    help="无台词也出料（只为后续 ASR 回标准备 wav）")
    ap.add_argument("--report", default="", help="可选：把 build.json 也写到这里")
    args = ap.parse_args(argv)

    if not args.voice_dir and not args.lines:
        print(json.dumps({"ok": False, "error": "至少需要 --voice-dir 或 --lines"}, ensure_ascii=False))
        return 2

    wav_dir = os.path.join(args.out, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    jobs = plan_jobs(args)
    if not jobs:
        print(json.dumps({
            "ok": False,
            "error": "没有匹配到任何语料；检查 --speaker 拼写或 --voice-dir 内容",
            "hint": "voice-only 模式要求一级子目录即角色，或显式给 --speaker",
        }, ensure_ascii=False))
        return 3

    stats = {
        "ok": True, "out": os.path.abspath(args.out), "wav_dir": os.path.abspath(wav_dir),
        "list": os.path.abspath(os.path.join(args.out, "list.txt")),
        "sample_rate": args.sample_rate, "lang": args.lang, "lang_auto": bool(args.lang_auto),
        "min_sec": args.min_sec, "max_sec": args.max_sec, "trim": not args.no_trim,
        "tail_pad": args.tail_pad, "trim_tail": bool(args.trim_tail),
        "speakers": [], "built": 0, "seconds": 0.0,
        "skipped_short": 0, "skipped_long": 0, "skipped_error": 0, "skipped_notext": 0,
        "skipped_missing": 0,
        "rejected_long": [], "short_samples": [], "errors": [],
        "cut_tail": [], "tail_end_db": [],
        "duration_histogram": {},
    }
    list_lines = []
    used_names = set()
    kept_durs = []

    for sp_name, payload in jobs:
        st = {"speaker": sp_name, "candidates": len(payload), "built": 0, "seconds": 0.0,
              "skipped_short": 0, "skipped_long": 0, "skipped_error": 0, "skipped_notext": 0,
              "skipped_missing": 0}
        n_done = 0
        seen_src = set()
        for src, text, scene in payload:
            if args.limit and n_done >= args.limit:
                break
            if not src:
                st["skipped_missing"] += 1     # 有台词但没有对应语音文件
                continue
            if src in seen_src:
                continue
            seen_src.add(src)

            if not text and not args.allow_empty_text:
                st["skipped_notext"] += 1
                continue
            if text and len(text) < args.min_chars:
                st["skipped_notext"] += 1
                continue
            if text and not is_meaningful(text):
                st["skipped_notext"] += 1
                continue

            stem = safe_name(f"{sp_name}_{scene}_{os.path.splitext(os.path.basename(src))[0]}")
            # 同名冲突会让后一条覆盖前一条，且 list.txt 里出现重复 wav 名
            base_stem, k = stem, 2
            while stem in used_names:
                stem = f"{base_stem}_{k}"
                k += 1

            dst = os.path.join(wav_dir, stem + ".wav")
            ok, dur, why = copy_or_convert(
                args.ffmpeg, src, dst, args.sample_rate, not args.no_trim, args.min_sec, args.max_sec,
                tail_pad=args.tail_pad, trim_tail=args.trim_tail
            )
            if not ok:
                if "短于下限" in why:
                    st["skipped_short"] += 1
                    if len(stats["short_samples"]) < 20:
                        stats["short_samples"].append(f"{os.path.basename(src)} {dur:.2f}s")
                elif "超长" in why:
                    st["skipped_long"] += 1
                    if len(stats["rejected_long"]) < 30:
                        stats["rejected_long"].append(f"{os.path.basename(src)} {dur:.1f}s")
                else:
                    st["skipped_error"] += 1
                    if len(stats["errors"]) < 10:
                        stats["errors"].append(f"{os.path.basename(src)}: {why}")
                continue

            used_names.add(stem)
            # 尾音体检：>-12dB 说明句子停在满电平（被切断），不是自然衰减
            el = end_loudness(dst)
            if el is not None:
                stats["tail_end_db"].append(round(el, 1))
                if el > -12:
                    stats["cut_tail"].append(f"{stem}.wav {el:.1f}dB")
            lang = guess_lang(text, args.lang) if args.lang_auto and text else args.lang
            list_lines.append(f"{stem}.wav|{sp_name}|{lang}|{text}")
            kept_durs.append(dur)
            st["built"] += 1
            st["seconds"] = round(st["seconds"] + dur, 2)
            n_done += 1

        st["minutes"] = round(st["seconds"] / 60.0, 2)
        stats["speakers"].append(st)
        stats["built"] += st["built"]
        stats["seconds"] = round(stats["seconds"] + st["seconds"], 2)
        for k in ("skipped_short", "skipped_long", "skipped_error", "skipped_notext", "skipped_missing"):
            stats[k] += st[k]

    stats["minutes"] = round(stats["seconds"] / 60.0, 2)

    # 尾音体检汇总：这是「切句尾」的唯一可判读信号，别只看句末有没有标点
    te = sorted(stats["tail_end_db"])
    if te:
        nn2 = len(te)
        stats["tail_stats"] = {
            "n": nn2,
            "median_db": round(te[nn2 // 2], 1),
            "cut_ratio": round(len(stats["cut_tail"]) / float(nn2), 3),
            "verdict": ("句尾被切断：训练出来的模型会吞末字" if len(stats["cut_tail"]) / float(nn2) > 0.3
                        else "正常（句尾有自然衰减）"),
        }
    stats["cut_tail"] = stats["cut_tail"][:20]

    # 时长直方图：让「下限该设多少」变成可判读的，而不是拍脑袋
    buckets = [(0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 8), (8, 10), (10, 15), (15, 999)]
    hist = {}
    for lo, hi in buckets:
        key = f"{lo:g}~{hi:g}s" if hi < 999 else f"{lo:g}s+"
        hist[key] = sum(1 for d in kept_durs if lo <= d < hi)
    stats["duration_histogram"] = hist
    if kept_durs:
        sd = sorted(kept_durs)
        nn = len(sd)
        stats["duration_stats"] = {
            "min": round(sd[0], 2),
            "median": round(sd[nn // 2], 2),
            "p90": round(sd[int(nn * 0.9) if nn > 1 else 0], 2),
            "max": round(sd[-1], 2),
            "avg": round(sum(sd) / nn, 2),
        }
    # 丢弃量远大于出料量时给出可执行建议（不自动改配置，避免静默）
    if stats["skipped_short"] > max(20, stats["built"]):
        stats["advice"] = (
            f"短句丢弃 {stats['skipped_short']} 条 > 出料 {stats['built']} 条。"
            f"当前下限 {args.min_sec}s；可降低 minSec 重跑（建议 1.5~2.0），"
            f"或用 tts_scan 看整体时长分布再决定"
        )

    if not list_lines:
        print(json.dumps({**stats, "ok": False,
                          "error": "一条都没出料。若语料本来就没台词，用 --allow-empty-text 出 wav 再走 ASR 回标"
                          }, ensure_ascii=False))
        return 4

    list_path = os.path.join(args.out, "list.txt")
    with open(list_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(list_lines) + "\n")

    stats["lines"] = len(list_lines)
    report_path = args.report or os.path.join(args.out, "build.json")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
