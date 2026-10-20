#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
番剧音轨「语音活动剖面」——用来定位 OP / ED / 片尾废料。

为什么需要它：番剧源常带完整 OP、ED、次回预告、赞助商静止画面，
这些既不是角色台词，还会让 ASR 产生歌词幻觉。定位后交给
`anime_lines.py run --keep` 跳过。

判据（不依赖 ASR，几十秒出结果）：
  - OP/ED 是**连续歌声**：VAD 段很长、段间停顿短而规整
  - 对白段：段短、停顿多、语音占比中等
  - 片尾废料：长时间无语音，或极短感叹词零星分布

用法：
    python profile_episode.py audio.m4a
    python profile_episode.py audio.m4a --block 20
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import anime_lines as AL


def rms_profile(audio: np.ndarray, sr: int, win: float = 1.0) -> np.ndarray:
    n = max(1, int(win * sr))
    m = len(audio) // n
    if m == 0:
        return np.zeros(1, dtype=np.float32)
    fr = audio[: m * n].reshape(m, n).astype(np.float32)
    return np.sqrt((fr ** 2).mean(axis=1))


def pct(v: float) -> str:
    return f"{v * 100:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--block", type=float, default=30.0, help="剖面统计块长（秒）")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--work", default=None, help="中间 wav 存放目录")
    args = ap.parse_args()

    ffmpeg = AL.find_ffmpeg(args.ffmpeg)
    src = os.path.abspath(args.input)
    # 注意：必须按输入文件名分目录，否则同一批多集会共用同一个
    # audio_16k.wav 缓存，第二集直接读到第一集的音频（实测踩过）。
    stem = os.path.splitext(os.path.basename(src))[0]
    work = args.work or os.path.join(os.path.dirname(src), "_profile", stem)
    os.makedirs(work, exist_ok=True)
    wav = os.path.join(work, "audio_16k.wav")
    if not os.path.isfile(wav):
        AL.log("提取 16k 分析轨 ...")
        AL.to_wav(src, wav, ffmpeg, 16000)

    sr, audio = AL.read_wav_mono(wav)
    total = len(audio) / sr
    AL.log(f"时长 {timedelta(seconds=int(total))}  ({total:.1f}s)")

    AL.log("VAD ...")
    regions = AL.detect_speech(audio, sr, pad_ms=120)
    AL.log(f"      {len(regions)} 段，语音合计 {sum(r.dur for r in regions):.1f}s "
           f"({sum(r.dur for r in regions) / total * 100:.1f}%)")

    # 每块：语音占比 / 平均能量 / 最长连续语音 / 最长静音
    nb = int(np.ceil(total / args.block))
    print()
    print(f"{'区间':>17} {'语音占比':>8} {'能量':>8} {'最长段':>7} {'最长静音':>9}  形态")
    print("-" * 74)
    prof = []
    for b in range(nb):
        t0 = b * args.block
        t1 = min(total, t0 + args.block)
        if t1 - t0 < 1.0:
            continue
        inner = [r for r in regions if r.end > t0 and r.start < t1]
        cover = sum(max(0.0, min(r.end, t1) - max(r.start, t0)) for r in inner)
        ratio = cover / (t1 - t0)
        seg = audio[int(t0 * sr): int(t1 * sr)]
        e = float(np.sqrt((seg.astype(np.float32) ** 2).mean())) if seg.size else 0.0
        longest_seg = max((min(r.end, t1) - max(r.start, t0) for r in inner), default=0.0)
        # 块内最长静音
        marks = [(max(r.start, t0), min(r.end, t1)) for r in inner]
        marks.sort()
        gap, cur = 0.0, t0
        for a, z in marks:
            if a - cur > gap:
                gap = a - cur
            cur = max(cur, z)
        gap = max(gap, t1 - cur)

        if ratio > 0.92 and longest_seg > 15:
            shape = "连续歌声? (OP/ED)"
        elif ratio < 0.05:
            shape = "近乎无语音 (废料/静止画面?)"
        elif gap > 15:
            shape = "大段静音"
        elif ratio > 0.6:
            shape = "密集对话"
        else:
            shape = "对白"
        prof.append((t0, t1, ratio, e, longest_seg, gap, shape))
        print(f"{timedelta(seconds=int(t0))!s:>7}-{timedelta(seconds=int(t1))!s:<9} "
              f"{pct(ratio):>8} {e:8.4f} {longest_seg:6.1f}s {gap:8.1f}s  {shape}")

    # 汇总：疑似 OP/ED 连续块
    print()
    print("== 疑似连续歌声区（OP/ED 候选）==")
    cand, run_ = [], []
    for row in prof:
        if row[6].startswith("连续歌声"):
            run_.append(row)
        else:
            if len(run_) >= 2:
                cand.append((run_[0][0], run_[-1][1]))
            run_ = []
    if len(run_) >= 2:
        cand.append((run_[0][0], run_[-1][1]))
    if cand:
        for a, z in cand:
            print(f"  {timedelta(seconds=int(a))} - {timedelta(seconds=int(z))}  "
                  f"({z - a:.0f}s)")
    else:
        print("  （未检出；OP/ED 可能带垫乐对白或音量较低，需人工听开头/结尾各 3 分钟）")

    print()
    print("== 首尾空白 ==")
    head = next((i for i, r in enumerate(prof) if r[2] > 0.10), None)
    tail = next((i for i, r in enumerate(reversed(prof)) if r[2] > 0.10), None)
    if head is not None:
        print(f"  开头 {prof[head][0]:.0f}s 之前几乎没有语音")
    if tail is not None:
        t = prof[len(prof) - 1 - tail][1]
        print(f"  结尾 {t:.0f}s 之后几乎没有语音（末尾静音 {total - t:.0f}s）")
    print()
    print("提示：把正片段落传给 run 的 --keep 即可，例如")
    print('  --keep "0:20-1:35,3:10-21:30"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
