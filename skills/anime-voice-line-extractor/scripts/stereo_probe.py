#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用「立体声侧信号比」定位 OP / ED / BGM 段。

原理：番剧对白几乎都中置（mono），左右声道相位一致 → 侧信号 L-R 接近 0；
OP/ED/配乐是立体声混音，声像铺开 → 侧信号明显。

同时给出低频能量占比（音乐鼓点重）作辅证。

用法：
    python stereo_probe.py audio.m4a --start 60 --end 260 --step 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anime_lines as AL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--ffmpeg", default=None)
    args = ap.parse_args()

    ffmpeg = AL.find_ffmpeg(args.ffmpeg)
    src = os.path.abspath(args.input)
    stem = os.path.splitext(os.path.basename(src))[0]
    work = os.path.join(os.path.dirname(src), "_profile", stem)
    os.makedirs(work, exist_ok=True)
    wav = os.path.join(work, f"stereo_{args.sr}.wav")
    if not os.path.isfile(wav):
        AL.log("提取立体声分析轨 ...")
        cmd = [ffmpeg, "-y", "-v", "error", "-i", src, "-ac", "2",
               "-ar", str(args.sr), "-vn", wav]
        subprocess.run(cmd, check=True)

    import soundfile as sf
    x, sr = sf.read(wav, always_2d=True, dtype="float32")
    total = len(x) / sr
    end = min(args.end or total, total)

    # 逐窗
    win = int(args.step * sr)
    i0, i1 = int(args.start * sr), int(end * sr)
    print(f"{'时间':>10} {'侧比':>7} {'低频占比':>8} {'能量':>8}   判断")
    print("-" * 58)
    t = args.start
    while i0 + win <= i1:
        blk = x[i0:i0 + win]
        L, R = blk[:, 0], blk[:, 1]
        mid = (L + R) * 0.5
        side = (L - R) * 0.5
        e_mid = float(np.sqrt((mid ** 2).mean())) + 1e-9
        e_side = float(np.sqrt((side ** 2).mean()))
        sratio = e_side / e_mid
        # 低频占比
        sp = np.abs(np.fft.rfft(mid * np.hanning(len(mid))))
        fr = np.fft.rfftfreq(len(mid), 1.0 / sr)
        low = sp[fr < 200].sum() / (sp.sum() + 1e-9)
        e = float(np.sqrt((mid ** 2).mean()))

        if sratio > 0.30:
            verdict = "立体声混音 (OP/ED?)"
        elif sratio > 0.18:
            verdict = "宽声场 (BGM 垫乐?)"
        else:
            verdict = "中置对白"
        print(f"{timedelta(seconds=int(t))!s:>10} {sratio:7.3f} {low:8.3f} {e:8.4f}   {verdict}")
        i0 += win
        t += args.step
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
