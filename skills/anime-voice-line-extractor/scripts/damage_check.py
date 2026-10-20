#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
damage_check.py —— 检出人声分离引入「显著损伤」的切片。

判据全部是「干声 vs 同一条原混音」的相对比较，自成参考，不需要干净样本：

 1. flat_ratio   语音帧谱平坦度比值（干声 / 原混音）。分离伪影是宽带白噪性质的，
                 会同时抬高所有 bin，平坦度上升；真语音的谐波结构平坦度很低。
 2. hnr_dry      浊音帧自相关峰值中位（谐噪比）。伪影破坏周期性 → HNR 掉。
 3. env_corr     20ms 包络相关（对齐后）。掉说明时域被削掉一块。
 4. speech_keep  语音帧（原混音能量前 30%）的能量保留 dB。
 5. clip_ratio   削波样本占比 + 直流偏移。

输出 damage.csv，含各项原始值与综合分 score（0~1，越高越差）、flags。
"""
import argparse
import csv
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.stats import pearsonr

SR = 44100
WIN = int(0.032 * SR)      # 32ms 分析窗
HOP = int(0.010 * SR)      # 10ms 步进


def frames(x, win=WIN, hop=HOP):
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def mono_read(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        x = resample_poly(x, SR, sr).astype(np.float32)
    return x


def spec_stats(x):
    """返回 (语音帧下标, 谱平坦度数组, 帧能量)"""
    F = frames(x) * np.hanning(WIN)[None, :]
    P = np.abs(np.fft.rfft(F, axis=1)) ** 2 + 1e-12
    # 只看 100Hz~8kHz，避开低频隆隆与超声段
    lo = int(100 / (SR / WIN))
    hi = int(8000 / (SR / WIN))
    Pb = P[:, lo:hi]
    flat = np.exp(np.mean(np.log(Pb), axis=1)) / np.mean(Pb, axis=1)
    energy = np.sqrt(np.mean(F ** 2, axis=1) + 1e-12)
    return flat, energy


def autocorr_peak(x):
    """归一化自相关在 80~500Hz 对应 lag 上的峰值，取浊音帧中位。"""
    F = frames(x) * np.hanning(WIN)[None, :]
    X = np.fft.rfft(F, n=2 * WIN, axis=1)
    ac = np.fft.irfft(np.abs(X) ** 2, axis=1)[:, :WIN]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(SR / 500), int(SR / 80)
    seg = ac[:, lo:hi]
    peak = seg.max(axis=1)
    energy = np.sqrt(np.mean(F ** 2, axis=1) + 1e-12)
    # 只在响亮的帧上取（静音帧自相关没意义）
    thr = np.percentile(energy, 60)
    m = energy >= thr
    return float(np.median(peak[m])) if m.any() else 0.0


def env20(x):
    f = int(0.02 * SR)
    n = len(x) // f
    if n < 8:
        return np.array([np.sqrt(np.mean(x ** 2) + 1e-12)])
    b = x[: n * f].reshape(n, f)
    return np.sqrt((b ** 2).mean(1) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="干声产物根目录（含 lines.csv）")
    ap.add_argument("--out", default="", help="输出 csv 路径")
    args = ap.parse_args()

    out_csv = args.out or os.path.join(args.root, "damage.csv")
    rows = list(csv.DictReader(open(os.path.join(args.root, "lines.csv"), encoding="utf-8-sig")))

    res = []
    for k, r in enumerate(rows, 1):
        dry_rel = r["dry_file"].replace("\\", "/")
        dry_p = os.path.join(args.root, dry_rel)
        mix_p = os.path.join(args.root, r["ep"], "_mix", os.path.basename(dry_rel))
        if not (os.path.isfile(dry_p) and os.path.isfile(mix_p)):
            print(f"  ! 跳过 {r['ep']}#{r['index']}（缺文件）", flush=True)
            continue

        d = mono_read(dry_p)
        m = mono_read(mix_p)
        n = min(len(d), len(m))
        d, m = d[:n], m[:n]
        if n < int(0.6 * SR):
            continue

        fd, ed = spec_stats(d)
        fm, em = spec_stats(m)

        # 语音帧：以原混音能量为准（原混音含 BGM，但 BGM 平稳，取前 30% 基本落在语音上）
        thr = np.percentile(em, 70)
        mask = em >= thr
        flat_d = float(np.median(fd[mask])) if mask.any() else float(np.median(fd))
        flat_m = float(np.median(fm[mask])) if mask.any() else float(np.median(fm))
        flat_ratio = flat_d / (flat_m + 1e-12)

        hnr = autocorr_peak(d)

        # 包络相关（先粗对齐 ±120ms）
        a, b = env20(d), env20(m)
        k = min(len(a), len(b))
        a, b = a[:k], b[:k]
        best, bl = -2.0, 0
        for L in range(-6, 7):
            aa = a[max(0, L): k + min(0, L)]
            bb = b[max(0, -L): k - max(0, L)]
            if len(aa) < 8:
                continue
            c = pearsonr(aa, bb)[0]
            if c > best:
                best, bl = c, L
        aa = a[max(0, bl): k + min(0, bl)]
        bb = b[max(0, -bl): k - max(0, bl)]
        kk = min(len(aa), len(bb))
        aa, bb = aa[:kk], bb[:kk]
        env_corr = float(pearsonr(aa, bb)[0]) if kk > 8 else 0.0

        # 语音帧能量保留：取原混音能量前 30% 的帧
        t2 = np.percentile(bb, 70)
        sel = bb >= t2
        keep_db = float(20 * np.log10((aa[sel].mean() + 1e-9) / (bb[sel].mean() + 1e-9))) if sel.any() else 0.0

        clip = float(np.mean(np.abs(d) > 0.995))
        dc = float(abs(np.mean(d)))

        res.append(dict(
            ep=r["ep"], index=r["index"], text=r.get("text", ""),
            dur=float(r["dur_dry"]), flat_ratio=round(flat_ratio, 3),
            hnr=round(hnr, 3), env_corr=round(env_corr, 3),
            keep_db=round(keep_db, 2), clip=round(clip, 5), dc=round(dc, 5),
            dry=dry_rel,
        ))
        if k % 25 == 0:
            print(f"  ...{k}/{len(rows)}", flush=True)

    # ---- 综合损伤分：各项按稳健 z 分数（中位/MAD）合并，越大越差
    def z(key, invert=False):
        v = np.array([r[key] for r in res], dtype=float)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826
        if mad < 1e-6:
            mad = max(np.std(v), 1e-6)
        zz = (v - med) / mad
        return -zz if invert else zz

    zf = z("flat_ratio")
    zh = z("hnr", invert=True)
    ze = z("env_corr", invert=True)
    zk = z("keep_db", invert=True)
    score = np.nan_to_num(0.40 * zf + 0.25 * zh + 0.20 * ze + 0.15 * zk)
    for r, s in zip(res, score):
        r["score"] = round(float(s), 3)

    res.sort(key=lambda r: -r["score"])
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)

    arr = {k: np.array([r[k] for r in res], dtype=float) for k in
           ("flat_ratio", "hnr", "env_corr", "keep_db", "score")}
    print(f"\n共 {len(res)} 条 → {out_csv}")
    for k in ("flat_ratio", "hnr", "env_corr", "keep_db"):
        v = arr[k]
        print(f"  {k:11s} p5={np.percentile(v,5):7.3f} p25={np.percentile(v,25):7.3f} "
              f"中位={np.median(v):7.3f} p75={np.percentile(v,75):7.3f} p95={np.percentile(v,95):7.3f}")
    print(f"\n损伤分最高 15 条：")
    for r in res[:15]:
        print(f"  {r['score']:6.2f} {r['ep']}#{r['index']:<4} {r['dur']:.2f}s "
              f"flat={r['flat_ratio']:.2f} hnr={r['hnr']:.2f} env={r['env_corr']:.2f} "
              f"keep={r['keep_db']:+.1f}dB  {r['text'][:20]}")


if __name__ == "__main__":
    sys.exit(main())
