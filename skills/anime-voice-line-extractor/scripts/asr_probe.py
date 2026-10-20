#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""转写指定区间，用来肉眼判定 OP / ED / 预告 / 长尾。

不做 VAD 过滤、不做幻觉拦截——目的就是看清"这段到底是什么"。
用法：
    python asr_probe.py audio.m4a --start 0 --end 360
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anime_lines as AL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--language", default="ja")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    ffmpeg = AL.find_ffmpeg(args.ffmpeg)
    src = os.path.abspath(args.input)
    stem = os.path.splitext(os.path.basename(src))[0]
    work = args.work or os.path.join(os.path.dirname(src), "_profile", stem)
    os.makedirs(work, exist_ok=True)
    wav = os.path.join(work, "audio_16k.wav")
    if not os.path.isfile(wav):
        AL.to_wav(src, wav, ffmpeg, 16000)

    sr, audio = AL.read_wav_mono(wav)
    total = len(audio) / sr
    end = min(args.end or total, total)

    model_dir = AL.find_whisper_model(None)
    tr = AL.Transcriber(model_dir, device="cuda", compute_type="int8_float16")
    AL.log(f"转写 {timedelta(seconds=int(args.start))} - "
           f"{timedelta(seconds=int(end))} ...")
    segs, info = tr.model.transcribe(
        wav, language=args.language, beam_size=5,
        vad_filter=False, word_timestamps=False,
        condition_on_previous_text=False,
        clip_timestamps=[args.start, end],
    )
    n = 0
    for s in segs:
        t = (s.text or "").strip()
        if not t:
            continue
        n += 1
        nsp = float(getattr(s, "no_speech_prob", 0.0) or 0.0)
        lp = float(getattr(s, "avg_logprob", 0.0) or 0.0)
        print(f"[{s.start:8.2f}-{s.end:8.2f}] nsp={nsp:.2f} lp={lp:6.2f}  {t}")
    AL.log(f"共 {n} 段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
