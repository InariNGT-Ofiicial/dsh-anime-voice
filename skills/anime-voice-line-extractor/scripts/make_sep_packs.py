# -*- coding: utf-8 -*-
"""把多条台词「打包」成接近 20 秒的长文件，供 MSST 逐包分离。

为什么要打包：bs_roformer 的推理窗口是固定的 `audio.chunk_size`（20 秒），
一条 3 秒的台词照样要跑满一整个窗口 —— 实测单条约 10 秒，250 条要 40 分钟。
把 4~5 条拼进一个窗口，速度提升 4~5 倍，而拼接点落在每条前后各 PAD 秒的
上下文余量里，成品切片时会被裁掉，不会进入训练音频。

用法：
  python make_sep_packs.py --lines <lines.csv> --mix <整集 44.1k wav> \
      --prefix ep01 --out <输出目录> --manifest <清单 json 追加写入> [--pad 0.4]

清单结构：{"<ep>|<切片文件名>": {"pack": "<包名>", "off": <采样点>, "n": <采样点数>}}
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import soundfile as sf

SR = 44100
MAX_SAMPLES = 860_000     # 略低于 882000（20s），保证一包只占一个推理窗口


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pad", type=float, default=0.4)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.lines, encoding="utf-8-sig")))
    os.makedirs(a.out, exist_ok=True)
    pad = int(a.pad * SR)

    man = {}
    if os.path.isfile(a.manifest):
        man = json.load(open(a.manifest, encoding="utf-8"))

    with sf.SoundFile(a.mix) as f:
        total = f.frames
        # 先算出每条的「带余量区间」
        spans = []
        for r in rows:
            s0, e0 = int(float(r["start"]) * SR), int(float(r["end"]) * SR)
            ps, pe = max(0, s0 - pad), min(total, e0 + pad)
            spans.append(dict(row=r, s0=s0, e0=e0, ps=ps, pe=pe, n=e0 - s0))

        # 贪心打包
        packs, cur, cur_len = [], [], 0
        for sp in spans:
            ln = sp["pe"] - sp["ps"]
            if cur and cur_len + ln > MAX_SAMPLES:
                packs.append(cur)
                cur, cur_len = [], 0
            cur.append(sp)
            cur_len += ln
        if cur:
            packs.append(cur)

        for pi, pack in enumerate(packs):
            name = f"{a.prefix}_p{pi:03d}"
            buf, off = [], 0
            pieces = []
            for sp in pack:
                f.seek(sp["ps"])
                x = f.read(sp["pe"] - sp["ps"], dtype="float32", always_2d=True)
                pieces.append(x)
                man[f"{a.prefix}|{os.path.basename(sp['row']['file'])}"] = dict(
                    pack=name, off=off + (sp["s0"] - sp["ps"]), n=sp["n"])
                off += x.shape[0]
            y = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 2), np.float32)
            with sf.SoundFile(os.path.join(a.out, name + ".wav"), "w", SR, 2,
                              subtype="PCM_16") as o:
                o.write(y)

    json.dump(man, open(a.manifest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{a.prefix}: {len(rows)} 条 → {len(packs)} 包，清单累计 {len(man)} 项")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
