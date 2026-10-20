#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
asr_regress.py —— 用「干声重转写 vs 原文本」判定分离是否损伤了语音。

损伤的直接证据只有三种，全部可观测：
  - 整段被判成「无语音」（no_speech 概率过高被丢）→ 人声基本没了
  - 转写文本显著短于原文（掉词）→ 局部被吃掉
  - 平均词概率显著下降 → 模糊/伪影盖住了发音

同音替换（単身不妊中 → 単身赴任中）CER 会升高，但音频本身没问题，
不属于损伤，所以主判据用「长度比 + 置信度」，CER 只作辅助参考。
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anime_lines import Transcriber, find_whisper_model  # noqa: E402

PUNCT = "、。，．,.!！?？「」『』（）()【】〔〕…・♪〜～:：;；\"'“”‘’「」ー 　"


def norm(s: str) -> str:
    s = (s or "").strip()
    for ch in PUNCT:
        s = s.replace(ch, "")
    return s


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--whisper-model", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.root, "lines.csv"), encoding="utf-8-sig")))
    if args.limit:
        rows = rows[: args.limit]
    out_csv = args.out or os.path.join(args.root, "asr_regress.csv")

    tr = Transcriber(find_whisper_model(args.whisper_model or None), device="cuda")
    res = []
    for k, r in enumerate(rows, 1):
        dry_p = os.path.join(args.root, r["dry_file"].replace("\\", "/"))
        if not os.path.isfile(dry_p):
            continue
        try:
            words, info = tr.transcribe(dry_p, language="ja")
        except Exception as e:
            print(f"  ! {r['ep']}#{r['index']}: {e}", flush=True)
            continue
        hyp = "".join(w["text"] for w in words)
        conf = sum(w["conf"] for w in words) / len(words) if words else 0.0
        ref = norm(r["text"])
        hn = norm(hyp)
        cer = lev(ref, hn) / max(1, len(ref))
        res.append(dict(
            ep=r["ep"], index=r["index"], dur=float(r["dur_dry"]),
            ref_len=len(ref), hyp_len=len(hn),
            len_ratio=round(len(hn) / max(1, len(ref)), 3),
            conf=round(conf, 4), old_conf=round(float(r.get("asr_conf") or 0), 4),
            conf_drop=round(float(r.get("asr_conf") or 0) - conf, 4),
            cer=round(cer, 3), dropped=int(getattr(tr, "last_dropped", 0)),
            hyp=hyp, ref=r["text"],
            dry=r["dry_file"].replace("\\", "/"),
        ))
        if k % 20 == 0:
            print(f"  ...{k}/{len(rows)}", flush=True)

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)

    import numpy as np
    lr = np.array([x["len_ratio"] for x in res])
    cd = np.array([x["conf_drop"] for x in res])
    ce = np.array([x["cer"] for x in res])
    print(f"\n共 {len(res)} 条 → {out_csv}")
    for nm, v in (("len_ratio", lr), ("conf_drop", cd), ("cer", ce)):
        print(f"  {nm:10s} p1={np.percentile(v,1):7.3f} p5={np.percentile(v,5):7.3f} "
              f"中位={np.median(v):7.3f} p95={np.percentile(v,95):7.3f}")
    print(f"\n最可疑 20 条：")
    bad = sorted(res, key=lambda x: (x["len_ratio"], -x["conf_drop"]))[:20]
    for x in bad:
        print(f"  {x['ep']}#{x['index']:<4} {x['dur']:5.2f}s len={x['len_ratio']:.2f} "
              f"conf={x['conf']:.2f}(原{x['old_conf']:.2f}) cer={x['cer']:.2f}  "
              f"{x['ref'][:16]} → {x['hyp'][:16]}")


if __name__ == "__main__":
    sys.exit(main())
