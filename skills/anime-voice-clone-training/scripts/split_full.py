# -*- coding: utf-8 -*-
"""split_full.py —— 整集正片 → 按句末标点切句 → 从连续干声切片（新主路线）。

和旧 pack 路线的三条本质区别：
  1. 输入是**整条正片**，不是 5 分钟 pack —— 不存在 pack 边界切断句子
  2. 断句**只认标点**（。！？…），不认停顿 —— 停顿是表演手段，不是句界
  3. 切片从**连续干声**取，句尾是自然衰减，不是人工淡出到零

断点优先级：句末标点 > 、（断出完整短语）> 组内最大停顿 > 硬切。
句尾定位 `tail_point()`：从词尾往后走，落到「本句 90 分位电平 −40dB」处再 +60ms。
文件命名 `<ep>_<4位序号>.wav` 扁平唯一 —— GPT-SoVITS 全流程拿 basename 当 key。

两步：
  --plan   只算句子规划（不需要干声，ASR 完成就能看规模）
  --slice  按规划从干声切片并导出 32k（需要 _full/voc/<ep>/vocals.wav）
"""
import argparse
import csv
import io
import json
import os
import re
import sys
from difflib import SequenceMatcher

import numpy as np

DEFAULT_BASE = os.environ.get("ANIME_WORK", os.getcwd())

ENDERS = "。！？!?…"
CLOSERS = "」』）)】"
MAX_DUR = 10.5          # 一句话最长（GPT-SoVITS 训练建议 3~10s）
MAX_CHAR = 45
MIN_DUR = 1.00
MIN_CHAR = 4
TAIL_MAX = 0.50         # 句尾最多接多少自然衰减
HEAD = 0.06             # 句首留一点起音

# 去 OP/ED 后的正片区间（绝对秒，与 _full/raw/<ep>.wav 的时间轴一致：
# raw 截到正片结尾，OP/ED 段留在文件里，靠这里的区间丢句）。
KEEP = {}  # Each project must provide its own reviewed time spans.


# ------------------------------------------------------------------ 断句
def group_sentences(words):
    """按句末标点成句；超长时优先在「、」处断，其次最大停顿。"""
    sents = []
    cur = []

    def flush(force=False):
        if not cur:
            return
        a, b = cur[0]["start"], cur[-1]["end"]
        txt = "".join(w["text"] for w in cur).strip().rstrip("、，,")
        sents.append(dict(words=list(cur), start=a, end=b, text=txt, forced=force))
        cur.clear()          # 必须清空：不清就会把整句再累积进下一句（句子层层嵌套）

    for w in words:
        cur.append(w)
        t = w["text"].rstrip()
        span = w["end"] - cur[0]["start"]
        chars = len("".join(x["text"] for x in cur).strip())
        ended = t.endswith(tuple(ENDERS)) or (t and t[-1] in CLOSERS
                                              and len(t) > 1 and t[-2] in ENDERS)
        if ended:
            flush(False)
            continue
        if span > MAX_DUR or chars > MAX_CHAR:
            # 断点优先级：逗号 > 组内最大停顿 > 硬切。
            # 在「、」处断出来的仍是完整短语，在词中间断出来的是半句。
            bi = None
            for i in range(len(cur) - 1, max(0, len(cur) // 3), -1):
                if cur[i - 1]["text"].rstrip().endswith(("、", "，", ",")):
                    bi = i
                    break
            if bi is None:
                best, bi2 = -1.0, len(cur) - 1
                for i in range(1, len(cur)):
                    g = cur[i]["start"] - cur[i - 1]["end"]
                    if g > best:
                        best, bi2 = g, i
                if bi2 < len(cur) - 1 and best > 0.12:
                    bi = bi2
            if bi is not None and 0 < bi < len(cur):
                head, tail = cur[:bi], cur[bi:]
                cur = head
                flush(True)
                cur = tail
            else:
                flush(True)
    flush(False)
    return sents


# ------------------------------------------------------------------ 切片
def tail_point(x, sr, i0, i1):
    """从句尾往后走，找到能量真正衰减下去的位置（自然收尾），不超过 i1。"""
    n = int(0.02 * sr)
    if i1 - i0 < n * 2:
        return i1
    hop = max(1, n)
    k = (i1 - i0) // hop
    seg = x[i0:i0 + k * hop]
    rms = np.sqrt((seg.reshape(k, hop) ** 2).mean(1) + 1e-12)
    ref = np.percentile(rms, 90)
    thr = ref * 10 ** (-40 / 20)          # 落到本句 90 分位的 −40dB 就算收完
    for j in range(k - 1, max(0, k // 6), -1):
        if rms[j] > thr:
            return i0 + min(i1 - i0, (j + 1) * hop + int(0.06 * sr))
    return i1


def slice_ep(ep, sents, keep, full, out):
    import soundfile as sf
    from scipy.signal import resample_poly

    voc_path = os.path.join(full, "voc", ep, "vocals.wav")
    if not os.path.isfile(voc_path):
        raise SystemExit(f"缺干声 {voc_path}")
    voc, sr = sf.read(voc_path, dtype="float32", always_2d=True)
    voc = voc.mean(1) if voc.shape[1] > 1 else voc[:, 0]
    # 扁平目录 + 唯一名：每集都从 0001 重新编号会让特征互相覆盖（丢 58% 的老坑）
    dst_dir = os.path.join(out, "wavs")
    os.makedirs(dst_dir, exist_ok=True)

    rows = []
    drop = dict(keep=0, short=0, silent=0, dur=0)
    for s in sents:
        if not (keep[0] <= s["start"] and s["end"] <= keep[1]):
            drop["keep"] += 1
            continue
        i0 = max(0, int(max(keep[0], s["start"] - HEAD) * sr))
        j1 = min(len(voc), int(min(keep[1], s["end"] + TAIL_MAX) * sr))
        # 不许越过下一句的起点
        j1 = min(j1, int((s["next_start"] - 0.03) * sr)) if s.get("next_start") else j1
        if j1 - i0 < int(0.5 * sr):
            drop["short"] += 1
            continue
        j = tail_point(voc, sr, int(s["end"] * sr), j1)
        seg = voc[i0:j]
        if len(seg) / sr < MIN_DUR:
            drop["dur"] += 1
            continue
        y = resample_poly(seg.astype(np.float64), 32000, sr).astype(np.float32)
        pk = float(np.max(np.abs(y)))
        if pk < 1e-4:
            drop["silent"] += 1
            continue
        name = f"{ep}_{s['gid']:04d}"
        sf.write(os.path.join(dst_dir, name + ".wav"), y, 32000)
        rows.append(dict(ep=ep, gid=s["gid"], name=name, start=round(s["start"], 3),
                         end=round(j / sr, 3), dur=round(len(y) / 32000, 3),
                         peak=round(pk, 3), text=s["text"], forced=s["forced"],
                         spk=s.get("spk", None)))
    print(f"  {ep} 切片 {len(rows)} 条 / 候选 {len(sents)} 句   丢弃明细 {drop}")
    return rows


def agree_with_plain(sent, plain_words):
    """与「整条一次」那版（asr_full.py）交叉校验。

    带提示词那版在**没有对白的音乐段**会编词（ep01 实测：冒出德语、串到别的番、反复
    「ご視聴ありがとうございました」）。同一时段让两版各说一遍，对不上就是编的。
    """
    lo, hi = sent["start"] - 0.15, sent["end"] + 0.15
    mid = [w["text"] for w in plain_words if w["end"] > lo and w["start"] < hi]
    if not mid:
        return 0.0
    a = re.sub(r"[\s。、，！？!?…「」『』（）()]", "", sent["text"])
    b = re.sub(r"[\s。、，！？!?…「」『』（）()]", "", "".join(mid))
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def parse_keep(spec: str):
    """'ep01=0-1747;ep02=0-99,171-1330' -> {ep: [(a,b), ...]}"""
    keep = {}
    for item in [x for x in spec.split(";") if x.strip()]:
        ep, spans = item.split("=", 1)
        keep[ep.strip()] = sorted((float(a), float(b))
                                  for a, b in (p.split("-", 1) for p in spans.split(",")))
    return keep


def main():
    ap = argparse.ArgumentParser(description="整集正片按标点切句（新主路线）")
    ap.add_argument("--base", default=DEFAULT_BASE, help="工作区根目录（含 _full）")
    ap.add_argument("--eps", default="ep01,ep02,ep03", help="逗号分隔")
    ap.add_argument("--keep", default="", help="覆盖 OP/ED 区间：'ep02=0-99,171-1330'，分号分隔多集")
    ap.add_argument("--role", required=True, help="train.list 的角色字段")
    ap.add_argument("--lang", default="ja", help="train.list 的语言字段")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--slice", action="store_true")
    a = ap.parse_args()

    full = os.path.join(a.base, "_full")
    out = os.path.join(full, "out")
    keep_map = dict(KEEP)
    if a.keep:
        keep_map.update(parse_keep(a.keep))
    sys.path.insert(0, os.environ.get("ANIME_EXTRACTOR_SCRIPTS", str(__import__("pathlib").Path(__file__).resolve().parents[2] / "anime-voice-line-extractor/scripts")))
    os.makedirs(os.path.join(out, "ts"), exist_ok=True)

    plans = {}
    for ep in [e.strip() for e in a.eps.split(",") if e.strip()]:
        pp = os.path.join(full, "asr", ep + "_p.json")     # 分块版带标点，优先
        p = pp
        if not os.path.isfile(p):
            raise SystemExit(f"缺少主 ASR: {p}")
        words = json.load(open(p, encoding="utf-8"))["words"]
        plain_p = os.path.join(full, "asr", ep + ".json")
        if not os.path.isfile(plain_p):
            raise SystemExit(f"缺少交叉校验 ASR: {plain_p}")
        plain = json.load(open(plain_p, encoding="utf-8"))["words"] \
            if os.path.isfile(plain_p) and plain_p != p else []

        sents = group_sentences(words)
        n0 = len(sents)
        if ep not in keep_map:
            raise SystemExit(f"为 {ep} 指定 --keep，不能套用其它作品区间")
        kp = keep_map[ep]
        sents = [s for s in sents
                 if any(x <= s["start"] and s["end"] <= y for x, y in kp)]
        n1 = len(sents)
        for s in sents:
            s["agree"] = agree_with_plain(s, plain) if plain else 1.0
        sents = [s for s in sents if s["agree"] >= 0.45]
        n2 = len(sents)

        for i, s in enumerate(sents):
            s["gid"] = i + 1
            s["next_start"] = sents[i + 1]["start"] if i + 1 < len(sents) else None
        plans[ep] = sents
        n = len(sents)
        if not n:
            raise SystemExit(f"{ep}: no sentences survived filtering")
        dur = np.array([s["end"] - s["start"] for s in sents])
        ended = sum(1 for s in sents if s["text"].rstrip().endswith(tuple(ENDERS)))
        forced = sum(1 for s in sents if s["forced"])
        ok = ((dur >= MIN_DUR) & (dur <= MAX_DUR))
        print(f"{ep}: 原 {n0} 句 → 去 OP/ED {n1} → 两版一致 {n2}")
        print(f"      时长 中位{np.median(dur):.2f}s  句末有标点 {ended}/{n} ({ended/n*100:.0f}%)  "
              f"强制断 {forced}  落在 {MIN_DUR}~{MAX_DUR}s 的 {ok.sum()} 条 "
              f"合计 {dur[ok].sum():.0f}s", flush=True)

    if a.plan and not a.slice:
        print("\n（只做规划，未切片）")
        return

    all_rows = []
    for ep, sents in plans.items():
        for span in keep_map[ep]:
            selected = [s for s in sents if span[0] <= s['start'] and s['end'] <= span[1]]
            if selected:
                all_rows += slice_ep(ep, selected, span, full, out)

    if not all_rows:
        raise SystemExit("没有切片产出")

    with io.open(os.path.join(out, "lines.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    dst = os.path.join(out, "train.list")
    with io.open(dst, "w", encoding="utf-8", newline="\n") as f:
        for r in all_rows:
            f.write(f"{os.path.join(out, 'wavs', r['name'] + '.wav')}|{a.role}|{a.lang}|{r['text']}\n")
    print(f"\n合计 {len(all_rows)} 条，输出 {out}\n待过声纹：{dst}")


if __name__ == "__main__":
    main()
