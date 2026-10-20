#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把已切好的台词切片换成「干声」（去 BGM / 音效 / 环境声）。

思路：分离在**整集**上做（上下文完整、伪影少），再按 `lines.csv` 里记录的
绝对时间轴从**人声轨**重切——不改任何一句的归属判定，只换音频源。

用法：
  python dry_vocals.py --vocals-dir <MSST输出目录> \
      --ep "ep01|lines.csv|mix.wav|ep01" \
      --ep "ep02|lines.csv|mix.wav|ep02" \
      --out <输出目录>

每个 --ep 的字段：子集名|原 lines.csv|原混音整集 wav|MSST 输出里的文件名主干
（MSST 产物名形如 `<stem>_vocals.wav` / `<stem>_other.wav`）

产出：
  <out>/<子集>/roles/<角色>/*.wav   干声（44.1k / mono / 16bit）
  <out>/lines.csv                   带质检列的总表
  <out>/train.list                  GPT-SoVITS 清单（指向干声）
  <out>/待校对.tsv                   人工校对表（BOM）
  <out>/质检.csv                     逐条客观指标
  <out>/A-B试听.html                原声/干声对照试听
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
import wave

import numpy as np

SR = 44100
FADE = 0.008          # 首尾淡入淡出（秒）
EDGE_MARGIN = 0.06    # 端点静音裁剪后保留的余量（秒）
EDGE_RATIO = 0.035    # 判定有声帧的相对门限（相对切片峰值帧）
TARGET_RMS = 0.10     # 目标有效值（≈ -20 dBFS），让 250 条切片响度一致
PEAK_CEIL = 0.89      # 峰值上限（≈ -1.0 dBFS）
MAX_GAIN_DB = 12.0
MIN_GAIN_DB = -10.0


# ---------------------------------------------------------------- 读写
def read_audio(path: str):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    return a, sr


def write_wav_mono(path: str, audio: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    a = np.clip(audio, -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- 基础度量
def frame_rms(x: np.ndarray, sr: int, win: float = 0.02) -> np.ndarray:
    f = max(1, int(win * sr))
    n = len(x) // f
    if n <= 0:
        return np.zeros(1, dtype=np.float32)
    b = x[: n * f].reshape(n, f).astype(np.float32)
    return np.sqrt((b ** 2).mean(axis=1) + 1e-12)


def noise_floor_ratio(x: np.ndarray, sr: int) -> float:
    """底噪相对峰值：非语音帧（10 分位 RMS）/ 峰值帧 RMS。越小越干净。"""
    r = frame_rms(x, sr)
    peak = float(r.max())
    if peak <= 1e-7:
        return -1.0
    return float(np.percentile(r, 10)) / peak


def band_energy(x: np.ndarray, sr: int, lo: float = 200.0, hi: float = 3800.0) -> float:
    """语音频带能量（单边谱积分）。"""
    if len(x) < 512:
        return 0.0
    w = np.hanning(len(x)).astype(np.float32)
    X = np.fft.rfft(x * w)
    p = (np.abs(X) ** 2)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    m = (freqs >= lo) & (freqs <= hi)
    return float(p[m].sum() + 1e-12)


def hf_ratio(x: np.ndarray, sr: int) -> float:
    """高频（>6kHz）能量占比，BGM 里的镲片/弦乐泛音多在这里暴露。"""
    if len(x) < 512:
        return 0.0
    w = np.hanning(len(x)).astype(np.float32)
    p = np.abs(np.fft.rfft(x * w)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    tot = float(p.sum()) + 1e-12
    return float(p[freqs > 6000].sum()) / tot


# ---------------------------------------------------------------- 切片
def to_mono(a: np.ndarray) -> tuple[np.ndarray, float]:
    """返回 (mono, 通道相关度)。相关度过低说明有反相成分，别直接平均。"""
    if a.shape[1] == 1:
        return a[:, 0].copy(), 1.0
    l, r = a[:, 0], a[:, 1]
    den = float(np.sqrt((l ** 2).sum() * (r ** 2).sum())) + 1e-12
    corr = float((l * r).sum()) / den
    if corr >= 0.3:
        return ((l + r) * 0.5).astype(np.float32), corr
    # 相关度低就用能量大的那只声道
    return (l if (l ** 2).sum() >= (r ** 2).sum() else r).copy(), corr


def trim_edges(seg: np.ndarray, sr: int) -> tuple[int, int]:
    """裁掉首尾静音，保留 EDGE_MARGIN 余量。返回 (start, end) 采样点。"""
    r = frame_rms(seg, sr)
    peak = float(r.max())
    if peak <= 1e-7:
        return 0, len(seg)
    voiced = np.where(r > peak * EDGE_RATIO)[0]
    if voiced.size == 0:
        return 0, len(seg)
    f = max(1, int(0.02 * sr))
    m = int(EDGE_MARGIN * sr)
    s = max(0, int(voiced[0]) * f - m)
    e = min(len(seg), (int(voiced[-1]) + 1) * f + m)
    return s, e


def apply_fade(seg: np.ndarray, sr: int) -> np.ndarray:
    nf = max(1, int(FADE * sr))
    if seg.size > 2 * nf:
        ramp = np.linspace(0.0, 1.0, nf, dtype=np.float32)
        seg = seg.copy()
        seg[:nf] *= ramp
        seg[-nf:] *= ramp[::-1]
    return seg


def normalize_level(seg: np.ndarray) -> tuple[np.ndarray, float]:
    """按有效值（RMS）拉齐响度，同时不让峰值顶到 0dBFS。

    为什么不用纯峰值归一：TTS 训练集里各条峰值主要取决于有没有爆破音，
    峰归一会让轻声、慢速的句子被抬得过高，模型学到的动态范围就乱了。
    """
    if seg.size == 0:
        return seg, 1.0
    rms = float(np.sqrt((seg.astype(np.float64) ** 2).mean()))
    peak = float(np.abs(seg).max())
    if rms <= 1e-7 or peak <= 1e-7:
        return seg, 1.0
    g_rms = TARGET_RMS / rms
    g_pk = PEAK_CEIL / peak
    gain = min(g_rms, g_pk)
    gain = float(np.clip(gain, 10 ** (MIN_GAIN_DB / 20), 10 ** (MAX_GAIN_DB / 20)))
    return (seg * gain).astype(np.float32), gain


# ---------------------------------------------------------------- 主流程
def parse_ep(spec: str) -> dict:
    parts = spec.split("|")
    if len(parts) != 4:
        raise SystemExit(f"--ep 格式错误：{spec!r}（应形如 ep01|lines.csv|mix.wav|ep01）")
    ep, lines, mix, stem = (p.strip() for p in parts)
    return dict(ep=ep, lines=lines, mix=mix, stem=stem)


def read_lines(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_stem(out_dir: str, stem: str, kind: str) -> str:
    cands = [
        os.path.join(out_dir, stem, f"{kind}.wav"),        # MSST 默认：<stem>/vocals.wav
        os.path.join(out_dir, stem, f"{kind}.flac"),
        os.path.join(out_dir, f"{stem}_{kind}.wav"),
        os.path.join(out_dir, f"{stem}_{kind}.flac"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    # 兜底：模糊匹配
    for root, _, files in os.walk(out_dir):
        for f in files:
            if kind in f and f.lower().endswith((".wav", ".flac")) and stem in root:
                return os.path.join(root, f)
    raise SystemExit(f"在 {out_dir} 找不到 {stem} 的 {kind} 产物")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocals-dir", required=True, help="MSST 输出目录")
    ap.add_argument("--ep", action="append", required=True,
                    help='子集|lines.csv|混音整集wav|MSST文件名主干')
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-mode", action="store_true",
                    help="干声是按包分离的（vocals-dir/<包名>/vocals.wav + 清单定位）")
    ap.add_argument("--packs-manifest", default="",
                    help="make_sep_packs.py 产出的 packs.json")
    ap.add_argument("--cut-pad", type=float, default=0.06,
                    help="切片两端各留的余量秒数（与原管线 slice_audio 的 pad 对齐）")
    ap.add_argument("--no-normalize", action="store_true", help="不做响度归一")
    ap.add_argument("--edge-trim", action="store_true",
                    help="裁掉首尾静音（默认保留原始 VAD 边界；裁了容易把 2s 级台词裁到门槛以下）")
    args = ap.parse_args()

    eps = [parse_ep(s) for s in args.ep]
    os.makedirs(args.out, exist_ok=True)

    all_rows, qc_rows = [], []
    ab_items = []
    per_ep: dict[str, list[dict]] = {e["ep"]: [] for e in eps}
    manifest, handles = {}, {}
    if args.clip_mode:
        if not args.packs_manifest or not os.path.isfile(args.packs_manifest):
            raise SystemExit("--clip-mode 需要 --packs-manifest（make_sep_packs.py 生成）")
        manifest = json.load(open(args.packs_manifest, encoding="utf-8"))

    for spec in eps:
        ep = spec["ep"]
        lines = read_lines(spec["lines"])
        import soundfile as sf
        mix_f = sf.SoundFile(spec["mix"])
        if mix_f.samplerate != SR:
            raise SystemExit(f"{ep}: 混音采样率 {mix_f.samplerate} != {SR}")
        total_mix = mix_f.frames

        if args.clip_mode:
            print(f"[{ep}] 逐条干声模式；切片 {len(lines)} 条", flush=True)
        else:
            vpath = find_stem(args.vocals_dir, spec["stem"], "vocals")
            voc_f = sf.SoundFile(vpath)
            if voc_f.samplerate != SR:
                raise SystemExit(f"{ep}: 干声轨采样率 {voc_f.samplerate} != {SR}")
            print(f"[{ep}] 干声轨 {os.path.basename(vpath)}；切片 {len(lines)} 条", flush=True)

        drop_cnt = 0
        corr_ep = 1.0

        for row in lines:
            start, end = float(row["start"]), float(row["end"])
            s = max(0, int(start * SR))
            e = min(total_mix, int(end * SR))
            if e - s < int(0.2 * SR):
                continue
            # 切片两端各留一点余量（与原管线 slice_audio 的 pad=0.06 对齐，
            # 这样很多 1.9s 的台词不会因为少这 0.12s 掉到 2s 门槛以下）
            mp = int(args.cut_pad * SR)
            cs = max(0, s - mp)
            ce = min(total_mix, e + mp)

            mix_f.seek(cs)
            seg_m_raw = mix_f.read(ce - cs, dtype="float32", always_2d=True)

            if args.clip_mode:
                # 逐包分离：多条拼成一个推理窗口，成品按清单里的采样点切回来
                key = f"{ep}|{os.path.basename(row['file'])}"
                ent = manifest.get(key)
                if not ent:
                    print(f"  ! 清单里没有 {key}", flush=True)
                    continue
                p = os.path.join(args.vocals_dir, ent["pack"], "vocals.wav")
                vf = handles.get(p)
                if vf is None:
                    if not os.path.isfile(p):
                        print(f"  ! 缺干声包 {p}", flush=True)
                        continue
                    vf = sf.SoundFile(p)
                    handles[p] = vf
                vf.seek(int(ent["off"]) - (s - cs))
                seg_v_raw = vf.read(ce - cs, dtype="float32", always_2d=True)
            else:
                voc_f.seek(cs)
                seg_v_raw = voc_f.read(ce - cs, dtype="float32", always_2d=True)

            if seg_v_raw.size == 0 or seg_m_raw.size == 0:
                continue
            k = min(len(seg_v_raw), len(seg_m_raw))
            seg_v_raw, seg_m_raw = seg_v_raw[:k], seg_m_raw[:k]
            if k < int(0.2 * SR):
                continue

            seg_v, corr = to_mono(np.array(seg_v_raw, dtype=np.float32))
            seg_m, _ = to_mono(np.array(seg_m_raw, dtype=np.float32))
            corr_ep = corr

            # 端点静音裁剪（默认关：原切片已经过 VAD + 静音压缩，边界够紧，
            # 再裁会让大量 2s 左右的台词掉到 2s 门槛以下而浪费素材）
            if args.edge_trim:
                a, b = trim_edges(seg_v, SR)
            else:
                a, b = 0, len(seg_v)
            seg_v, seg_m = seg_v[a:b], seg_m[a:b]
            if seg_v.size < int(0.2 * SR):
                continue

            seg_v = apply_fade(seg_v, SR)
            seg_m = apply_fade(seg_m, SR)

            # ---- 质检：分离前后底噪 / 高频残留 / 语音频带保留率（都在归一之前算，
            #      否则归一带来的整体增益会污染比值）
            nf_v = noise_floor_ratio(seg_v, SR)
            nf_m = noise_floor_ratio(seg_m, SR)
            hf_v = hf_ratio(seg_v, SR)
            hf_m = hf_ratio(seg_m, SR)
            be_v = band_energy(seg_v, SR)
            be_m = band_energy(seg_m, SR)
            keep = be_v / be_m if be_m > 0 else 0.0
            if keep < 0.35:
                drop_cnt += 1

            if not args.no_normalize:
                seg_v, gain = normalize_level(seg_v)
            else:
                gain = 1.0

            speaker = row["speaker"]
            fname = os.path.basename(row["file"])
            rel = f"roles/{speaker}/{fname}"
            dst = os.path.join(args.out, ep, rel)
            write_wav_mono(dst, seg_v, SR)

            # 原切片复制一份到 _mix/，让 A/B 试听页自包含
            src_mix = os.path.join(os.path.dirname(os.path.abspath(spec["lines"])), row["file"])
            mix_rel = f"_mix/{fname}"
            dst_mix = os.path.join(args.out, ep, mix_rel)
            if os.path.isfile(src_mix) and not os.path.isfile(dst_mix):
                os.makedirs(os.path.dirname(dst_mix), exist_ok=True)
                with open(src_mix, "rb") as fi, open(dst_mix, "wb") as fo:
                    fo.write(fi.read())

            dur = len(seg_v) / SR
            qc = dict(
                ep=ep, index=row.get("index", ""), speaker=speaker,
                start=start, end=end, dur=round(dur, 3),
                floor_before=round(nf_m * 100, 2) if nf_m >= 0 else "",
                floor_after=round(nf_v * 100, 2) if nf_v >= 0 else "",
                hf_before=round(hf_m * 100, 2), hf_after=round(hf_v * 100, 2),
                band_keep=round(keep, 3), gain_db=round(20 * np.log10(gain), 1),
                chan_corr=round(corr, 3), peak=round(float(np.abs(seg_v).max()), 3),
                file=os.path.join(ep, rel).replace("\\", "/"),
            )
            qc_rows.append(qc)
            all_rows.append(dict(row, ep=ep, dry_file=qc["file"], dur_dry=round(dur, 3)))
            per_ep[ep].append(dict(
                index=row.get("index", ""), speaker=speaker, start=row["start"],
                end=row["end"], dur=round(dur, 3), text=row.get("text", ""),
                asr_conf=row.get("asr_conf", ""), sv_conf=row.get("sv_conf", ""),
                file=rel.replace("\\", "/")))

            ab_items.append(dict(
                ep=ep, speaker=speaker, text=row.get("text", ""), dur=dur,
                mix=f"{ep}/{mix_rel}",          # 浏览器里必须是正斜杠
                dry=qc["file"],
                floor_before=qc["floor_before"], floor_after=qc["floor_after"],
            ))

        print(f"[{ep}] 完成 {len([r for r in all_rows if r['ep'] == ep])} 条，"
              f"声道相关度 {corr_ep:.2f}，疑似误杀(频带保留<35%) {drop_cnt} 条", flush=True)
        mix_f.close()
        if not args.clip_mode:
            voc_f.close()

    # ---- 汇总表
    fields = ["ep", "index", "speaker", "start", "end", "dur_dry", "text",
              "asr_conf", "sv_conf", "dry_file"]
    with open(os.path.join(args.out, "lines.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # 每集单独一份（字段与 anime_lines 产物一致），便于 export_trainset 逐集跑
    ep_fields = ["index", "speaker", "start", "end", "dur", "text",
                 "asr_conf", "sv_conf", "file"]
    for ep, rows in per_ep.items():
        with open(os.path.join(args.out, ep, "lines.csv"), "w",
                  encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ep_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    with open(os.path.join(args.out, "质检.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(qc_rows[0].keys()))
        w.writeheader()
        w.writerows(qc_rows)

    total = sum(r["dur_dry"] for r in all_rows)
    with open(os.path.join(args.out, "train.list"), "w", encoding="utf-8") as f:
        for r in all_rows:
            p = os.path.join(args.out, r["dry_file"].replace("/", os.sep))
            f.write(f"{p}|{r['speaker']}|ja|{r['text']}\n")

    with open(os.path.join(args.out, "待校对.tsv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["文件", "角色", "文本", "时长", "底噪前%", "底噪后%"])
        for r, q in zip(all_rows, qc_rows):
            w.writerow([os.path.join(args.out, r["dry_file"].replace("/", os.sep)), r["speaker"],
                        r["text"], q["dur"], q["floor_before"], q["floor_after"]])

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(sr=SR, count=len(all_rows), total_sec=round(total, 1),
                       episodes=[e["ep"] for e in eps]), f, ensure_ascii=False, indent=2)

    write_report(args.out, ab_items, qc_rows, total)
    print(f"\n合计 {len(all_rows)} 条 / {total:.1f}s → {args.out}")
    return 0


def write_report(out_dir: str, items: list[dict], qc: list[dict], total: float) -> None:
    fb = [q["floor_before"] for q in qc if q["floor_before"] != ""]
    fa = [q["floor_after"] for q in qc if q["floor_after"] != ""]
    mb, ma = float(np.mean(fb)) if fb else 0.0, float(np.mean(fa)) if fa else 0.0
    kb = float(np.mean([q["band_keep"] for q in qc])) if qc else 0.0

    def esc(s):
        return html.escape(str(s))

    blocks = []
    for it in items:
        blocks.append(f"""
<article class="card" data-ep="{esc(it['ep'])}">
  <header><span class="tag">{esc(it['ep'])}</span><b>{esc(it['speaker'])}</b>
    <span class="dur">{it['dur']:.2f}s</span>
    <span class="nf">{esc(it['floor_before'])}% → <em>{esc(it['floor_after'])}%</em></span></header>
  <p class="txt">{esc(it['text'])}</p>
  <div class="row"><label>原声</label><audio controls preload="none" src="{esc(it['mix'])}"></audio></div>
  <div class="row"><label>干声</label><audio controls preload="none" src="{esc(it['dry'])}"></audio></div>
</article>""")

    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>久留美 干声 A/B 试听</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--dim:#6b7280;--line:#e5e7eb;--acc:#2563eb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14161a;--card:#1c1f25;--ink:#e8eaed;--dim:#9aa3ad;--line:#2c313a;--acc:#6ea8fe}}}}
*{{box-sizing:border-box}}
body{{margin:0;padding:28px 20px 60px;background:var(--bg);color:var(--ink);
font:15px/1.6 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}}
h1{{font-size:20px;margin:0 0 6px}}
.sub{{color:var(--dim);font-size:13px;margin-bottom:18px}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:120px}}
.stat b{{display:block;font-size:18px}}
.stat span{{color:var(--dim);font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}}
.card header{{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--dim)}}
.card header b{{color:var(--ink);font-size:13px}}
.tag{{background:var(--acc);color:#fff;border-radius:4px;padding:1px 6px;font-size:11px}}
.dur{{margin-left:auto}}
.nf em{{color:#16a34a;font-style:normal}}
.txt{{margin:8px 0 10px;font-size:14px;min-height:2.6em}}
.row{{display:flex;align-items:center;gap:8px;margin-top:6px}}
.row label{{width:36px;font-size:12px;color:var(--dim);flex:none}}
audio{{width:100%;height:32px}}
</style></head><body>
<h1>《FX 战士久留美》台词干声 A/B 对照</h1>
<div class="sub">共 {len(items)} 条 / {total/60:.1f} 分钟。每条上为原混音，下为分离后干声。</div>
<div class="stats">
  <div class="stat"><b>{len(items)}</b><span>切片数</span></div>
  <div class="stat"><b>{total/60:.1f} 分</b><span>总时长</span></div>
  <div class="stat"><b>{mb:.2f}% → {ma:.2f}%</b><span>底噪相对峰值（均值）</span></div>
  <div class="stat"><b>{kb*100:.0f}%</b><span>语音频带能量保留</span></div>
</div>
<div class="grid">{''.join(blocks)}</div>
</body></html>"""
    with open(os.path.join(out_dir, "A-B试听.html"), "w", encoding="utf-8") as f:
        f.write(doc)


if __name__ == "__main__":
    sys.exit(main())
