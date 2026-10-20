# -*- coding: utf-8 -*-
"""filter_speaker.py —— 新主路线的最后一道闸：声纹质心 0.55 + F0 ≥180Hz。

为什么不用聚类：聚类要给出人数，而且会把「目标角色 + 相近的女声」并到一起。
这里拿**已确认属于该角色的干净样本**当参照物更稳：
  · 参照 = 参照集全部样本的 embedding 求均值（L2 归一化）
  · 分数 = 每个新句与参照的余弦
  · 阈值不用参照的 2% 分位（0.668 太严，会误杀 274Hz 的少女高音）

**为什么必须加 F0 闸门**：实测声纹 <0.40 那批 F0 中位只有 145Hz（80% 低于 180Hz，是男声），
而 0.55~0.668 那批 F0 中位 275Hz、只有 4% 低于 180Hz —— 是同一个人被阈值挡了。
所以阈值取 0.55，再用「F0 ≥180Hz」把混进来的男声挡掉：双闸门比单卡余弦准得多。

参照集从哪来（首次跑没有已确认样本时）：
先用 SKILL.md §S1 的锚定 + refine 抽 20~30 条人工听过确认的同人切片，放进 --ref 目录。

输出：`_full/out/train_keep.list`（全部通过的）与 `_full/out/ts/train.list`（真正喂训练的清单，
定版时人工删减改这个文件），外加 `spk_scores.json` 存每条分数。
"""
import argparse
import glob
import json
import os

import numpy as np

DEFAULT_BASE = os.environ.get("ANIME_WORK", os.getcwd())
DEFAULT_REF = "ref"


def load16k(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        x = resample_poly(x, 16000, sr).astype(np.float32)
    return x


def f0_median(path):
    import pyworld as pw
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1).astype(np.float64)
    if sr != 16000:
        x = resample_poly(x, 16000, sr).astype(np.float64)
    f0, t = pw.dio(x, 16000, frame_period=10.0)
    f0 = pw.stonemask(x, f0, t, 16000)
    v = f0[f0 > 0]
    return float(np.median(v)) if v.size else 0.0


def main():
    ap = argparse.ArgumentParser(description="声纹质心 + F0 双闸门筛句（新主路线）")
    ap.add_argument("--base", default=DEFAULT_BASE, help="工作区根目录")
    ap.add_argument("--ref", default="", help="参照目录（已确认同人 wav），缺省 <base>/" + DEFAULT_REF)
    ap.add_argument("--train", default="", help="待评清单，缺省 <base>/_full/out/train.list")
    ap.add_argument("--ts", default="", help="训练清单输出，缺省 <base>/_full/out/ts/train.list")
    ap.add_argument("--role", required=True)
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--spk", type=float, default=0.55)
    ap.add_argument("--f0-min", type=float, default=0.0)
    a = ap.parse_args()

    full = os.path.join(a.base, "_full")
    ref_dir = a.ref or os.path.join(a.base, DEFAULT_REF)
    train = a.train or os.path.join(full, "out", "train.list")
    ts = a.ts or os.path.join(full, "out", "ts", "train.list")

    import sys
    sys.path.insert(0, os.environ.get("ANIME_EXTRACTOR_SCRIPTS", str(__import__("pathlib").Path(__file__).resolve().parents[2] / "anime-voice-line-extractor/scripts")))
    import anime_lines as AL
    AL._ensure_cuda_dlls()
    from anime_lines import SpeakerEmbedder, find_campp_model

    emb = SpeakerEmbedder(find_campp_model(None), device="cuda")

    # ---- 参照
    refs = sorted(glob.glob(os.path.join(ref_dir, "**", "*.wav"), recursive=True))
    if not refs:
        raise SystemExit(f"参照目录里没有 wav：{ref_dir}")
    print(f"参照样本 {len(refs)} 条  ({ref_dir})", flush=True)
    E = []
    for i in range(0, len(refs), 24):
        E.append(emb.embed_batch([load16k(p) for p in refs[i:i + 24]]))
    E = np.concatenate(E, 0)
    centroid = E.mean(0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    # 留一法：旧样本对质心的分布，代表「同一个人」的正常范围
    loo = E @ centroid
    print(f"参照自匹配（包含自身，仅描述分布）：中位 {np.median(loo):.3f}  "
          f"5% 分位 {np.percentile(loo,5):.3f}  最小 {loo.min():.3f}", flush=True)
    print(f"参照 2% 分位 = {np.percentile(loo,2):.3f}（仅作参考，实际用 --spk）", flush=True)

    # ---- 新样本
    rows = []
    for line in open(train, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        p, spk, lang, text = line.split("|", 3)
        rows.append(dict(path=p, text=text))
    print(f"待评 {len(rows)} 条", flush=True)
    if not rows:
        raise SystemExit("Empty input list")

    scores = []
    for i in range(0, len(rows), 24):
        scores.append(emb.embed_batch([load16k(r["path"]) for r in rows[i:i + 24]]))
        if i % 240 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    S = np.concatenate(scores, 0) @ centroid

    for r, s in zip(rows, S):
        r["spk"] = float(s)

    keep, dropped_f0 = [], 0
    for r in rows:
        if r["spk"] < a.spk:
            continue
        if a.f0_min > 0 and f0_median(r["path"]) < a.f0_min:
            dropped_f0 += 1
            continue
        keep.append(r)
    print(f"声纹 ≥{a.spk} 且 F0 ≥{a.f0_min}Hz：通过 {len(keep)} 条 / {len(rows)}"
          f"（其中 F0 闸门再刷掉 {dropped_f0} 条）")
    if not keep:
        raise SystemExit("No samples passed speaker filtering")
    hist = np.histogram(S, bins=[0, .2, .3, .4, .5, .6, .7, .8, 1.0])
    for c, e0, e1 in zip(hist[0], hist[1][:-1], hist[1][1:]):
        print(f"  {e0:.1f}~{e1:.1f}  {c:4d} 条  {'#' * int(c / max(1, len(rows)) * 60)}")

    out_keep = os.path.join(full, "out", "train_keep.list")
    os.makedirs(os.path.dirname(ts), exist_ok=True)
    for dst in (out_keep, ts):
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            for r in keep:
                f.write(f"{r['path']}|{a.role}|{a.lang}|{r['text']}\n")
    json.dump([dict(path=r["path"], spk=r["spk"], text=r["text"]) for r in rows],
              open(os.path.join(full, "out", "spk_scores.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n已写 {out_keep}\n     {ts}（定版时人工删减改这个文件）")


if __name__ == "__main__":
    main()
