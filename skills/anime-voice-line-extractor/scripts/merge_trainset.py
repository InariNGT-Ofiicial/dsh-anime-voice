#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把多集的 export_trainset 产物合并成一份训练集。

为什么需要单独一步：`export_trainset.py` 按 `角色_0001.wav` 命名，多集直接
输出到同一目录会互相覆盖，所以每集先落到自己的子目录，再由这里合并。

用法：
    python merge_trainset.py out/久留美_trainset_3ep --order ep01,ep02,ep03
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="包含各集子目录的根目录")
    ap.add_argument("--order", default="", help="合并顺序，逗号分隔；默认按名字排序")
    ap.add_argument("--out", default="", help="输出目录，默认用 root")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out or root)
    subs = [s.strip() for s in args.order.split(",") if s.strip()] or \
        sorted(d for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d))
               and os.path.isfile(os.path.join(root, d, "train.list")))
    subs = [s for s in subs if os.path.isfile(os.path.join(root, s, "train.list"))]
    if not subs:
        print(f"[x] {root} 下没有找到带 train.list 的子目录")
        return 2

    all_lines, rows = [], []
    total = 0.0
    print(f"{'集':<8}{'条数':>5}{'总时长':>10}{'均长':>8}")
    print("-" * 34)
    for s in subs:
        p = os.path.join(root, s, "train.list")
        lines = [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()]
        # 时长从 wav 头里读，避免依赖 tsv
        import wave
        durs = []
        for ln in lines:
            w = ln.split("|")[0]
            try:
                with wave.open(w, "rb") as f:
                    durs.append(f.getnframes() / f.getframerate())
            except Exception:
                durs.append(0.0)
        d = sum(durs)
        total += d
        print(f"{s:<8}{len(lines):>5}{d:>9.1f}s{(d / max(len(lines), 1)):>7.2f}s")
        all_lines.extend(lines)
        rows.append((s, lines, durs))

    print("-" * 34)
    print(f"{'合计':<8}{len(all_lines):>5}{total:>9.1f}s"
          f"{(total / max(len(all_lines), 1)):>7.2f}s  "
          f"= {timedelta(seconds=int(total))}")

    with open(os.path.join(out, "train.list"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")

    # 待校对.tsv：加一列「集」，方便按集过一遍
    with open(os.path.join(out, "待校对.tsv"), "w", encoding="utf-8-sig") as f:
        f.write("集\t序号\t时长\t字每秒\tASR\t风险\t文本\n")
        n = 0
        for s, lines, _ in rows:
            tp = os.path.join(root, s, "待校对.tsv")
            if not os.path.isfile(tp):
                continue
            with open(tp, encoding="utf-8-sig") as g:
                next(g, None)                     # 表头
                for ln in g:
                    ln = ln.rstrip("\n")
                    if not ln:
                        continue
                    n += 1
                    f.write(f"{s}\t{ln}\n")

    print()
    print(f"已合并 -> {os.path.join(out, 'train.list')}（{len(all_lines)} 行）")
    print(f"         {os.path.join(out, '待校对.tsv')}（{n} 行）")
    print()
    print("下一步：待校对.tsv 必须人耳过一遍——ASR 同音错字会原样教给模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
