#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_corpus.py —— 盘点语料：每个角色有多少条、多少秒、多少条带台词。

为什么要单独扫一遍：
    GPT-SoVITS 的成败几乎全在语料量与文本覆盖度上。
    「有 900 个语音文件」和「有 900 条可训数据」是两件事 ——
    前者可能在剧本里全都没标说话人，后者才决定能不能开训。
    这一步就是把这两个数字分开摆出来，避免训完才发现只有 12 条。

时长探测优先用 soundfile（快，venv 里有），失败回落 ffprobe（万能但每文件一个进程）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

AUDIO_EXT = (".wav", ".ogg", ".mp3", ".flac", ".opus", ".m4a", ".aac", ".wma")

try:
    import soundfile as sf
    _HAS_SF = True
except Exception:                                             # noqa: BLE001
    _HAS_SF = False


def dur_sf(path: str) -> float:
    try:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:                                         # noqa: BLE001
        return -1.0


def dur_ffprobe(ffprobe: str, path: str) -> float:
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, timeout=30,
        )
        return float((r.stdout or b"0").decode().strip() or 0)
    except Exception:                                         # noqa: BLE001
        return -1.0


def strip_index_prefix(s: str) -> str:
    t = re.sub(r"^[０-９0-9]{1,3}\s*[・･_.\-、，,．]\s*", "", s).strip()
    return t or s


def audio_files_under(d: str):
    out = []
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.lower().endswith(AUDIO_EXT):
                out.append(os.path.join(root, f))
    return sorted(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="盘点语料")
    ap.add_argument("--lines", default="")
    ap.add_argument("--voice-dir", default="")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--timeout", type=float, default=120.0, help="总扫描时长上限（秒），超时截断")
    ap.add_argument("--min-sec", type=float, default=3.0)
    ap.add_argument("--max-sec", type=float, default=10.0)
    args = ap.parse_args(argv)

    import time
    t0 = time.time()
    res = {"ok": True, "mode": "", "speakers": [], "has_soundfile": _HAS_SF}

    # ── 台词驱动 ──────────────────────────────────────────────────────────
    if args.lines and os.path.isfile(args.lines):
        rows = []
        with open(args.lines, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        idx = {}
        if args.voice_dir and os.path.isdir(args.voice_dir):
            for p in audio_files_under(args.voice_dir):
                idx.setdefault(re.sub(r"\.[^.]+$", "", os.path.basename(p)).lower(), p)
                idx.setdefault(os.path.basename(p).lower(), p)

        buckets = {}
        for r in rows:
            sp = str(r.get("speaker") or r.get("name") or "").strip()
            if not sp:
                continue
            v = r.get("voice") or r.get("vofile") or r.get("voice_id")
            src = ""
            if isinstance(v, str) and v:
                src = idx.get(re.sub(r"\.[^.]+$", "", v).lower()) or idx.get(os.path.basename(v).lower()) or ""
            b = buckets.setdefault(sp, {"speaker": sp, "lines": 0, "withVoice": 0,
                                        "withText": 0, "both": 0, "seconds": 0.0,
                                        "inRange": 0, "noDuration": 0})
            b["lines"] += 1
            has_text = bool(str(r.get("text") or r.get("message") or "").strip())
            if has_text:
                b["withText"] += 1
            if src:
                b["withVoice"] += 1
                if has_text:
                    b["both"] += 1
                if time.time() - t0 > args.timeout:
                    b["noDuration"] += 1
                    continue
                d = dur_sf(src) if _HAS_SF else -1.0
                if d < 0:
                    d = dur_ffprobe(args.ffprobe, src)
                if d < 0:
                    b["noDuration"] += 1
                else:
                    b["seconds"] = round(b["seconds"] + d, 2)
                    if args.min_sec <= d <= args.max_sec:
                        b["inRange"] += 1

        res["mode"] = "lines"
        res["speakers"] = sorted(buckets.values(), key=lambda x: -x["both"])
        res["truncated"] = (time.time() - t0) > args.timeout

    # ── 目录驱动 ──────────────────────────────────────────────────────────
    elif args.voice_dir and os.path.isdir(args.voice_dir):
        subs = sorted(
            d for d in (os.path.join(args.voice_dir, x) for x in os.listdir(args.voice_dir))
            if os.path.isdir(d)
        )
        items = []
        if subs:
            for d in subs:
                items.append((strip_index_prefix(os.path.basename(d)), audio_files_under(d)))
        else:
            items.append((strip_index_prefix(os.path.basename(args.voice_dir.rstrip("/\\"))),
                          audio_files_under(args.voice_dir)))

        for name, files in items:
            if not files:
                continue
            b = {"speaker": name, "lines": 0, "withVoice": len(files), "withText": 0,
                 "both": 0, "seconds": 0.0, "inRange": 0, "noDuration": 0}
            for p in files:
                if time.time() - t0 > args.timeout:
                    b["noDuration"] += 1
                    continue
                d = dur_sf(p) if _HAS_SF else -1.0
                if d < 0:
                    d = dur_ffprobe(args.ffprobe, p)
                if d < 0:
                    b["noDuration"] += 1
                else:
                    b["seconds"] = round(b["seconds"] + d, 2)
                    if args.min_sec <= d <= args.max_sec:
                        b["inRange"] += 1
            res["speakers"].append(b)
        res["mode"] = "voice-only"
        res["truncated"] = (time.time() - t0) > args.timeout
    else:
        print(json.dumps({"ok": False, "error": "需要 --lines 或 --voice-dir"}, ensure_ascii=False))
        return 2

    # ── 结论：能不能训 ───────────────────────────────────────────────────
    for b in res["speakers"]:
        b["minutes"] = round(b["seconds"] / 60.0, 1)
        usable = b.get("both") if res["mode"] == "lines" else b["inRange"]
        # 分档：<5min 基本没戏；5~20min 能听出人但不像；20~60min 可用；>60min 好
        if usable == 0:
            b["verdict"] = "无可用数据"
        elif b["minutes"] < 5:
            b["verdict"] = "太少（<5min）：音色会不稳"
        elif b["minutes"] < 20:
            b["verdict"] = "偏少（5~20min）：能出人声，辨识度有限"
        elif b["minutes"] < 60:
            b["verdict"] = "可用（20~60min）"
        else:
            b["verdict"] = "充足（>60min）"

    res["elapsedSec"] = round(time.time() - t0, 1)
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
