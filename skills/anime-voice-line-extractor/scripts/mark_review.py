#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mark_review.py —— 用「干声重转写」给训练集的待校对清单分级。

同音错字（単身不妊中 / 単身赴任中）人耳不看不出来，但训进去就是错的监督信号。
157 条全看一遍成本高，而转写差异大的只占一小部分：按字错率分级后，
只需要看「高」和「中」两档。
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_regress import norm, lev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="干声产物根目录（含 asr_regress.csv）")
    ap.add_argument("--trainset", default="", help="trainset 目录，默认 <root>/trainset")
    args = ap.parse_args()

    ts = args.trainset or os.path.join(args.root, "trainset")
    ar_rows = list(csv.DictReader(open(os.path.join(args.root, "asr_regress.csv"),
                                      encoding="utf-8-sig")))
    # 不能用「序号」匹配：待校对.tsv 的序号是训练集重排后的编号，与原始 index 不同。
    # 按（集, 归一化文本）配对，同文本多条时按出现顺序取。
    from collections import defaultdict
    pool = defaultdict(list)
    for a in ar_rows:
        pool[(a["ep"], norm(a["ref"]))].append(a)
    cursor = defaultdict(int)

    def take(ep, text):
        k = (ep, norm(text))
        lst = pool.get(k) or []
        i = cursor[k]
        if i < len(lst):
            cursor[k] = i + 1
            return lst[i]
        return None

    rows = list(csv.DictReader(open(os.path.join(ts, "待校对.tsv"), encoding="utf-8-sig"),
                               delimiter="\t"))

    # 把训练集里的实际文件路径补上（train.list 与待校对的文本一一对应）
    from collections import defaultdict as _dd
    fpool = _dd(list)
    for ln in open(os.path.join(ts, "train.list"), encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split("|")
        if len(parts) >= 4:
            fpool[norm(parts[3])].append(parts[0])
    fcur = _dd(int)

    def take_file(text):
        k = norm(text)
        lst = fpool.get(k) or []
        i = fcur[k]
        if i < len(lst):
            fcur[k] = i + 1
            return lst[i]
        return ""

    out = []
    for r in rows:
        r["文件"] = take_file(r["文本"])
        a = take(r["集"], r["文本"])
        if not a:
            r.update(干声转写="", 字错率="", 风险级别="未知")
            out.append(r)
            continue
        ref, hyp = norm(r["文本"]), norm(a["hyp"])
        cer = lev(ref, hyp) / max(1, len(ref))
        if cer >= 0.30:
            lvl = "高"
        elif cer >= 0.12:
            lvl = "中"
        else:
            lvl = "低"
        r.update(干声转写=a["hyp"], 字错率=f"{cer:.2f}", 风险级别=lvl)
        out.append(r)

    order = {"高": 0, "中": 1, "未知": 2, "低": 3}
    out.sort(key=lambda r: (order.get(r["风险级别"], 3), -float(r["字错率"] or 0)))

    cols = ["风险级别", "集", "序号", "时长", "文本", "干声转写", "字错率", "ASR", "风险", "字每秒", "文件"]
    p = os.path.join(ts, "待校对_分级.tsv")
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(out)

    from collections import Counter
    c = Counter(r["风险级别"] for r in out)
    print(f"→ {p}")
    print(f"高 {c['高']} 条 / 中 {c['中']} 条 / 低 {c['低']} 条"
          + (f" / 未知 {c['未知']} 条" if c.get("未知") else ""))
    print(f"必看 {c['高'] + c['中']} 条，占 {100*(c['高']+c['中'])/max(1,len(out)):.0f}%\n")
    for r in out[:22]:
        print(f"  [{r['风险级别']}] {r['集']}#{r['序号']:<4} cer={r['字错率']:<5} {r['文本']}")
        if r["干声转写"] and r["干声转写"] != r["文本"]:
            print(f"        → {r['干声转写']}")


if __name__ == "__main__":
    sys.exit(main())
