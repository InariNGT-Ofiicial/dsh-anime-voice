#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asr_label.py —— 给「只有语音、没有台词」的语料回标，把它变成 GPT-SoVITS 能训的数据集。

为什么需要它：
    DDSP/SVC 不需要文本，所以《千恋＊万花》那类 voice/<角色>/*.ogg 目录可以直接训；
    GPT-SoVITS 是 TTS，**没有文本就完全没有监督信号**。ASR 回标是这类语料唯一的路。

两条容易踩的：
    1. 一定要用 VAD 之外的整段转写：语料已经裁过静音，再让 whisper 自己 VAD 会切碎日文语气词。
    2. 置信度必须留痕。低置信度句子会直接污染音色（模型学到"含糊"，不是学到"清晰"），
       所以默认把 avg_logprob 低于阈值的整句剔掉，并把剔掉的清单写进报告，让人能复核。

只依赖 faster-whisper（装在 GPT-SoVITS 的 venv 里）。找不到模型时优先用本地缓存，
缓存也没有才允许联网下载 —— 沙箱里联网下载常常卡住，静默卡住比报错更浪费时间。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

BAR_RE = re.compile(r"[|｜]")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WS_RE = re.compile(r"[ \t\u3000]+")
MEANINGFUL_RE = re.compile(r"[\w\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", re.UNICODE)

AUDIO_EXT = (".wav", ".ogg", ".mp3", ".flac", ".opus", ".m4a")


def clean_text(s: str) -> str:
    if not s:
        return ""
    t = str(s).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    t = CTRL_RE.sub("", t)
    t = BAR_RE.sub("，", t)
    return WS_RE.sub(" ", t).strip()


def find_local_snapshot(cache_root: str, model: str):
    """
    HF 缓存里的 faster-whisper 模型落在
        <HF_HOME>/hub/models--Systran--faster-whisper-<model>/snapshots/<rev>/
    直接给路径最稳：省掉一次联网查询，也不会因为代理问题卡住。
    """
    if not cache_root:
        return ""
    hub = os.path.join(cache_root, "hub")
    cand = os.path.join(hub, f"models--Systran--faster-whisper-{model}", "snapshots")
    if not os.path.isdir(cand):
        return ""
    revs = [os.path.join(cand, x) for x in os.listdir(cand)]
    revs = [r for r in revs if os.path.isdir(r)]
    # 有 model.bin 的才算完整快照
    for r in sorted(revs, reverse=True):
        if os.path.isfile(os.path.join(r, "model.bin")):
            return r
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="faster-whisper 给语料回标台词")
    ap.add_argument("--wav-file", default="", help="单文件模式：只转写这一个文件，结果打到 stdout（不写 list.txt）")
    ap.add_argument("--wav-dir", default="", help="已规格化的 wav 目录")
    ap.add_argument("--out", default="", help="输出目录（写 list.txt 与 asr.json）")
    ap.add_argument("--speaker", default="speaker", help="写进 list.txt 的说话人名")
    ap.add_argument("--lang", default="auto", help="auto/zh/ja/en/ko/yue")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default="", help="HF_HOME；留空用环境变量")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--min-logprob", type=float, default=-1.0,
                    help="平均对数概率下限，低于此判为不可信并剔除")
    ap.add_argument("--max-no-speech", type=float, default=0.6,
                    help="静音概率上限，高于此判为不像人声")
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge-existing", default="", help="已有 list.txt：只补缺失的行（断点续跑）")
    ap.add_argument("--allow-download", action="store_true", help="本地缓存没有时允许联网下载")
    args = ap.parse_args(argv)

    # 缓存目录：显式参数 > 环境变量
    cache = args.cache_dir or os.environ.get("HF_HOME") or ""
    if cache:
        os.environ["HF_HOME"] = cache
        os.environ["HF_HUB_CACHE"] = os.path.join(cache, "hub")

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        print(json.dumps({"ok": False, "error": f"faster-whisper 未安装：{e}"}, ensure_ascii=False))
        return 2

    snap = find_local_snapshot(cache, args.model)
    if not snap and not args.allow_download:
        print(json.dumps({
            "ok": False,
            "error": f"本地缓存里没有 faster-whisper-{args.model}，且未允许下载",
            "cache": cache, "hint": "加 --allow-download，或先手动把模型放进 HF 缓存",
        }, ensure_ascii=False))
        return 3

    model_ref = snap or args.model

    def load(device, compute_type):
        return WhisperModel(model_ref, device=device, compute_type=compute_type,
                            download_root=None if snap else cache or None)

    # ★ ctranslate2 不认 torch 自带的 CUDA 运行时。本机实测：torch 是 cu128（自带
    #   cublas64_12.dll 在 torch/lib 下），但 ctranslate2 不会去那里找，加载 CuDNN/cuBLAS
    #   时报 "Library cublas64_12.dll is not found"。与其让用户去改 PATH，不如**自动降级 CPU**
    #   并把实际用的设备报出来 —— ASR 本来就只是兜底手段，慢一点可以接受，报错不行。
    used_device, used_type = args.device, args.compute_type
    try:
        model = load(args.device, args.compute_type)
    except Exception as e:                                    # noqa: BLE001
        first_err = e
        loaded = False
        # ① 同一设备换个精度再试一次（float16 在个别驱动上会挂）
        if args.compute_type == "float16" and args.device == "cuda":
            try:
                model = load(args.device, "int8_float16")
                used_type = "int8_float16"
                loaded = True
            except Exception as e2:                           # noqa: BLE001
                first_err = e2
        # ② 还不行且原本在 GPU 上 → 降 CPU
        if not loaded and args.device != "cpu":
            try:
                model = load("cpu", "int8")
                used_device, used_type = "cpu", "int8"
                loaded = True
            except Exception as e3:                           # noqa: BLE001
                print(json.dumps({"ok": False, "error": f"模型加载失败（含 CPU 降级）：{e3}"},
                                 ensure_ascii=False))
                return 4
        if not loaded:
            print(json.dumps({"ok": False, "error": f"模型加载失败：{first_err}"}, ensure_ascii=False))
            return 4
    if used_device != args.device:
        print(f"[asr] GPU 不可用（{args.device} → {used_device}），已降级 CPU；"
              f"原因通常是 ctranslate2 找不到 cuBLAS/cuDNN", file=sys.stderr)

    # ★ ctranslate2 是**惰性**加载 CUDA 库的：构造 WhisperModel 时可能一切正常，
    #   到第一次 transcribe 才抛 "Library cublas64_12.dll is not found"。
    #   所以降级逻辑必须包住转写调用，只包构造会漏。
    state = {"model": model, "device": used_device, "type": used_type}

    def ensure_cpu():
        if state["device"] == "cpu":
            return False
        try:
            state["model"] = load("cpu", "int8")
        except Exception:                                     # noqa: BLE001
            return False
        state["device"], state["type"] = "cpu", "int8"
        print("[asr] 转写时 CUDA 库加载失败，已切 CPU 重试（会慢，但能出结果）", file=sys.stderr)
        return True

    def transcribe(source, lang_hint):
        kw = dict(language=lang_hint, beam_size=args.beam_size,
                  vad_filter=False, condition_on_previous_text=False)
        try:
            segs, info = state["model"].transcribe(source, **kw)
            return list(segs), info
        except Exception as e:                                # noqa: BLE001
            if not ensure_cpu():
                raise
            segs, info = state["model"].transcribe(source, **kw)
            return list(segs), info

    # ── 单文件模式：给 tts_infer 做「合成完回读校验」用 ────────────────────
    if args.wav_file:
        if not os.path.isfile(args.wav_file):
            print(json.dumps({"ok": False, "error": f"文件不存在: {args.wav_file}"}, ensure_ascii=False))
            return 2
        lang_hint = None if args.lang == "auto" else args.lang
        try:
            segs, info = transcribe(args.wav_file, lang_hint)
        except Exception as e:                                # noqa: BLE001
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
            return 4
        text = clean_text("".join(s.text for s in segs))
        total = sum(max(1e-3, s.end - s.start) for s in segs) or 1e-3
        avg_lp = (sum(s.avg_logprob * max(1e-3, s.end - s.start) for s in segs) / total) if segs else 0.0
        lang = lang_hint or getattr(info, "language", None) or "zh"
        print(json.dumps({
            "ok": bool(text), "text": text, "lang": lang,
            "avgLogprob": round(avg_lp, 3), "segments": len(segs),
            "model": model_ref,
            "device": state["device"], "compute_type": state["type"],
            "deviceDowngraded": state["device"] != args.device,
        }, ensure_ascii=False))
        return 0 if text else 5

    if not args.wav_dir:
        print(json.dumps({"ok": False, "error": "需要 --wav-dir 或 --wav-file"}, ensure_ascii=False))
        return 2
    if not args.out:
        print(json.dumps({"ok": False, "error": "批量模式需要 --out"}, ensure_ascii=False))
        return 2

    wavs = sorted(
        os.path.join(args.wav_dir, f) for f in os.listdir(args.wav_dir)
        if f.lower().endswith(AUDIO_EXT)
    )
    if args.limit:
        wavs = wavs[: args.limit]

    done = {}
    if args.merge_existing and os.path.isfile(args.merge_existing):
        with open(args.merge_existing, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) == 4 and parts[3].strip():
                    done[parts[0]] = line.rstrip("\n")

    out_path = os.path.join(args.out, "list.txt")
    os.makedirs(args.out, exist_ok=True)

    rows, rejected, failed = [], [], []
    lang_hint = None if args.lang == "auto" else args.lang

    for i, w in enumerate(wavs, 1):
        name = os.path.basename(w)
        if name in done:
            rows.append(done[name])
            continue
        try:
            segs, info = transcribe(w, lang_hint)
        except Exception as e:                                # noqa: BLE001
            failed.append(f"{name}: {type(e).__name__}: {e}")
            continue

        text = clean_text("".join(s.text for s in segs))
        if not segs:
            rejected.append({"wav": name, "why": "no_segment"})
            continue
        # 平均对数概率按段时长加权，比取最小值稳（短段偶尔很低）
        total_dur = sum(max(1e-3, s.end - s.start) for s in segs)
        avg_lp = sum(s.avg_logprob * max(1e-3, s.end - s.start) for s in segs) / max(1e-6, total_dur)
        max_ns = max((getattr(s, "no_speech_prob", 0.0) or 0.0) for s in segs)

        if avg_lp < args.min_logprob:
            rejected.append({"wav": name, "why": "low_logprob", "avg_logprob": round(avg_lp, 3)})
            continue
        if max_ns > args.max_no_speech:
            rejected.append({"wav": name, "why": "no_speech", "no_speech_prob": round(max_ns, 3)})
            continue
        if len(text) < args.min_chars or not MEANINGFUL_RE.search(text):
            rejected.append({"wav": name, "why": "empty_text", "text": text[:40]})
            continue

        lang = lang_hint or getattr(info, "language", None) or "zh"
        # whisper 偶尔返回 zh 之外的标记，收口到 GPT-SoVITS 的白名单
        if lang not in ("zh", "ja", "en", "ko", "yue"):
            lang = {"chinese": "zh", "japanese": "ja", "english": "en", "korean": "ko"}.get(lang, "zh")
        rows.append(f"{name}|{args.speaker}|{lang}|{text}")

        if i % 20 == 0:
            print(f"[asr] {i}/{len(wavs)} rows={len(rows)} rejected={len(rejected)}", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n" if rows else "")

    stats = {
        "ok": bool(rows),
        "list": os.path.abspath(out_path),
        "wav_total": len(wavs),
        "labelled": len(rows),
        "rejected": len(rejected),
        "failed": len(failed),
        "model": model_ref,
        "device": state["device"],
        "compute_type": state["type"],
        "deviceDowngraded": state["device"] != args.device,
        "lang": args.lang,
        "rejected_samples": rejected[:30],
        "failed_samples": failed[:10],
    }
    with open(os.path.join(args.out, "asr.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if not rows:
        stats["error"] = "一条都没标出来：检查语言、模型与语料是否真的是人声"
    print(json.dumps(stats, ensure_ascii=False))
    return 0 if rows else 5


if __name__ == "__main__":
    sys.exit(main())
