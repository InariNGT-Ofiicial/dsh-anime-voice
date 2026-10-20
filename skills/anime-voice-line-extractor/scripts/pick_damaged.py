#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pick_damaged.py —— 从干声批次里挑出「显著损伤」的条目并剔除。

设计原则（踩过的坑）：
  声学偏离（包络相关、谱平坦度）在 BGM 强烈的段落会**系统性失效**——
  去掉 BGM 本来就会让干声包络与混音不符，这不叫损伤。EP01#76 相关内容
  全对、置信度 0.92，却因 env_corr=-0.11 被误判。
  所以主判据必须是「文本与音频是否对得上」，声学指标只作交叉验证。

四类显著损伤（任一命中即剔）：
  T1 内容缺失   干声转写显著短于原文（len_ratio ≤ 0.75）→ 语音被削掉一段
  T2 内容多出   干声转写显著长于原文（len_ratio ≥ 1.30）→ 原文本缺字，图不对文
  T3 听不清     置信度低且字错得多（conf < 0.55 且 CER ≥ 0.5）→ 伪影盖住发音
  T4 无声       转写完全为空 → 人声没了
  T5 时域崩塌   env_corr < 0.25 且 keep_db < -5（仅在内容也有缺失时才认）
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_regress import norm, lev  # noqa: E402


def load(path):
    if not os.path.isfile(path):
        raise SystemExit(f"缺文件：{path}")
    return {(r["ep"], r["index"]): r for r in
            csv.DictReader(open(path, encoding="utf-8-sig"))}


def classify(a, d):
    lr = float(a["len_ratio"])
    conf = float(a["conf"])
    cer = float(a["cer"])
    env = float(d.get("env_corr") or 0)
    keep = float(d.get("keep_db") or 0)
    ref = norm(a["ref"])
    hyp = norm(a["hyp"])

    if len(hyp) == 0:
        return "T4", "干声转写为空，人声丢失"
    if lr <= 0.75:
        return "T1", f"掉了 {100*(1-lr):.0f}% 内容（{a['ref']} → {a['hyp']}）"
    if lr >= 1.30:
        return "T2", f"音频比文本多 {100*(lr-1):.0f}%，原文本缺字（{a['ref']} → {a['hyp']}）"
    if conf < 0.55 and cer >= 0.5:
        return "T3", f"置信度 {conf:.2f} 且字错率 {cer:.0%}，发音被盖住"
    if env < 0.25 and keep < -5 and lr < 0.95:
        return "T5", f"时域崩塌（env={env:.2f}, 能量保留 {keep:+.1f}dB）"
    return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    dm = load(os.path.join(args.root, "damage.csv"))
    ar = load(os.path.join(args.root, "asr_regress.csv"))
    out_dir = args.out_dir or args.root

    kept, cut = [], []
    for k, a in sorted(ar.items()):
        d = dm.get(k, {})
        tag, why = classify(a, d)
        rec = dict(ep=k[0], index=k[1], dur=a["dur"], text=a["ref"],
                   asr_hyp=a["hyp"], len_ratio=a["len_ratio"],
                   conf=a["conf"], old_conf=a["old_conf"], cer=a["cer"],
                   env_corr=d.get("env_corr", ""), keep_db=d.get("keep_db", ""),
                   flat_ratio=d.get("flat_ratio", ""), hnr=d.get("hnr", ""),
                   dry=a["dry"], code=tag, reason=why)
        (cut if tag else kept).append(rec)

    fields = ["ep", "index", "dur", "text", "asr_hyp", "len_ratio", "conf",
              "old_conf", "cer", "env_corr", "keep_db", "flat_ratio", "hnr",
              "code", "reason", "dry"]
    with open(os.path.join(out_dir, "损伤剔除.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(cut)
    with open(os.path.join(out_dir, "保留清单.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)

    from collections import Counter
    c = Counter(r["code"] for r in cut)
    dur_cut = sum(float(r["dur"]) for r in cut)
    print(f"剔除 {len(cut)} 条 / {dur_cut:.1f}s，保留 {len(kept)} 条 / "
          f"{sum(float(r['dur']) for r in kept):.1f}s")
    print("分类：", dict(c))
    print()
    for r in cut:
        print(f"  [{r['code']}] {r['ep']}#{r['index']:<4} {float(r['dur']):5.2f}s  {r['reason']}")


if __name__ == "__main__":
    sys.exit(main())
