#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨集角色锚定：用已定案的某一集声纹中心，判定新一集每个簇是谁。

为什么需要：声纹聚类的簇编号每集独立重排（S01 只代表"本集戏份最多的那个"），
所以不能把上一集的 S01/S04 直接套到新一集。人的声纹本身是稳定的，
用余弦相似度比对簇中心即可客观映射，不用靠耳朵听。

用法：
    # 先看 EP01 的角色分布，确认哪些簇同属一人
    python anchor_speakers.py --ref ../out/FX久留美_EP01_final \
        --groups "久留美=S01,S04,S07,S09; 母亲=S03,S06; 父亲=S02,S05" \
        --target ../out/久留美_EP02
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np


def load_run(d: str):
    work = os.path.join(d, "_work")
    z = np.load(os.path.join(work, "sv_embeddings.npz"), allow_pickle=True)
    X, labels, durs = z["X"], z["labels"].astype(int), z["durs"]
    # 每个簇的时长与条数
    stats = {}
    for lb, du in zip(labels, durs):
        s = stats.setdefault(int(lb), [0, 0.0])
        s[0] += 1
        s[1] += float(du)
    return X, labels, durs, stats


def centroids(X, labels, min_dur: float = 0.0, durs=None):
    cents = {}
    for lb in sorted(set(labels.tolist())):
        m = labels == lb
        if durs is not None and float(durs[m].sum()) < min_dur:
            continue
        c = X[m].mean(axis=0)
        n = float(np.linalg.norm(c))
        cents[lb] = c / n if n > 0 else c
    return cents


def sid_of(lb: int) -> str:
    """0-based 簇号 -> 输出用的 S01 形式。"""
    return f"S{lb + 1:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="已定案的那一集（run 产物目录）")
    ap.add_argument("--groups", required=True,
                    help='角色=簇列表，用分号分隔：久留美=S01,S04; 母亲=S03')
    ap.add_argument("--target", required=True, help="要判定的新一集")
    ap.add_argument("--min-dur", type=float, default=0.8,
                    help="目标簇总时长低于此值不参与比对")
    args = ap.parse_args()

    Xr, lr, dr, sr = load_run(args.ref)
    cr = centroids(Xr, lr, durs=dr)

    groups = {}
    for part in args.groups.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, ids = part.split("=", 1)
        g = []
        for tok in ids.split(","):
            tok = tok.strip().upper()
            if not tok:
                continue
            if tok.startswith("S"):
                tok = tok[1:]
            g.append(int(tok) - 1)          # S01 -> 0
        groups[name.strip()] = g

    # 每个角色的参考中心 = 成员簇中心的平均
    refs = {}
    for name, ids in groups.items():
        vs = [cr[i] for i in ids if i in cr]
        if not vs:
            print(f"[warn] {name} 的簇在参考集里都不存在：{ids}")
            continue
        c = np.vstack(vs).mean(axis=0)
        n = float(np.linalg.norm(c))
        refs[name] = c / n if n > 0 else c

    Xt, lt, dt, st = load_run(args.target)
    ct = centroids(Xt, lt, min_dur=args.min_dur, durs=dt)

    names = list(refs.keys())
    R = np.vstack([refs[n] for n in names])
    tgt_ids = sorted(ct.keys(), key=lambda i: -st[i][1])

    print(f"参考：{os.path.basename(args.ref)}   目标：{os.path.basename(args.target)}")
    print(f"目标共 {len(tgt_ids)} 个簇（总时长 ≥ {args.min_dur}s）")
    print()
    head = f"{'簇':<6}{'条数':>5}{'时长':>8}  " + "".join(f"{n:>10}" for n in names) + "   判定"
    print(head)
    print("-" * (len(head) + 4))
    for lb in tgt_ids:
        v = ct[lb] @ R.T
        best = int(v.argmax())
        n_seg, dur = st[lb]
        row = f"{sid_of(lb):<6}{n_seg:>5}{dur:8.1f}s  " + \
              "".join(f"{x:10.3f}" for x in v)
        verdict = names[best] if float(v[best]) >= 0.45 else "?"
        print(f"{row}   {verdict}")
    print()
    print("判读：同一人的簇中心相似度通常 > 0.55；不同人 < 0.30。")
    print("      0.30~0.45 之间要结合台词内容人工确认（声线接近的母女/同声优）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
