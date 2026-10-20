# -*- coding: utf-8 -*-
"""
把 anime_lines 的产物转成 GPT-SoVITS 可训练的 list，并做质量筛除。

用法:
    python export_trainset.py <run_dir> [--src <run_dir_with_work>] [--out <dir>] [--dry]

筛除依据（实测得出）:
  1. 时长     —— 单条最佳 3-10s；<2s 时 GPT 阶段上下文不足
                （本机 yoshino 训练集：383 条/27.2 分，最短 2.01s，中位 3.81s）
  2. 语速     —— 字/秒 过低（长静音/标题卡）或过高（抢话/被压叠）
  3. ASR 置信 —— <0.80 的句子错字概率显著上升
  4. 跨词截断 —— 用词级时间戳精确判定：句首/句尾是否落在某个词的**内部**。
                这比字符启发式可靠（日语以「の」「て」结尾是正常的，不算截断）。
  5. 底噪     —— 帧能量低分位相对人声峰值的比例，提示 BGM/环境音
  6. 拼接修复 —— 相邻两条间隔 <0.25s 且断点两侧各是一个片假名单字 → 拼回
"""
import argparse
import csv
import json
import os
import re
import wave
from collections import Counter

import numpy as np

KATA = re.compile(r"^[\u30a0-\u30ff]$")


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return sr, x


def write_wav(path, x, sr):
    y = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((y * 32767).astype("<i2").tobytes())


def load_boundaries(src_dir):
    """返回词边界列表 [(start, end, text)]，用于判断切片是否切在词中间。"""
    p = os.path.join(src_dir, "_work", "words.json")
    if not os.path.isfile(p):
        return None
    ws = json.load(open(p, encoding="utf-8"))
    return [(float(w["start"]), float(w["end"]), w.get("text", "")) for w in ws]


def cut_inside_word(t, bounds, tol=0.06):
    """t 是否落在某个词的内部（离两边都超过 tol）。"""
    if not bounds:
        return False
    for s, e, _ in bounds:
        if s + tol < t < e - tol:
            return True
    return False


def n_chars(t):
    return len(re.sub(r"[\s、。,.!?！？…「」『』・ー…\-—]", "", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="被筛选的产物目录（含 lines.csv）")
    ap.add_argument("--src", default=None,
                    help="含 _work/words.json 的 run 目录（默认同 run_dir）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lang", default="", help="显式语言；缺省从 run/src 的 run_config.json 读取")
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--min-cps", type=float, default=2.5)
    ap.add_argument("--max-cps", type=float, default=9.5)
    ap.add_argument("--min-asr", type=float, default=0.80)
    ap.add_argument("--min-chars", type=int, default=6)
    ap.add_argument("--max-lowq-ratio", type=float, default=0.055,
                    help="低能量分位/峰值之比，超过视为底噪偏高（提示 BGM）")
    ap.add_argument("--keep-noisy", action="store_true", help="保留底噪偏高的条目")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--lines", default="",
                    help="替代 lines.csv 的清单文件（用于按剔除名单预过滤）")
    ap.add_argument("--file-field", default="",
                    help="文件路径字段名；默认自动识别 file / dry_file")
    ap.add_argument("--exclude", default="",
                    help="排除清单（含 ep,index 两列），剔除损伤条目时用")
    a = ap.parse_args()

    run = a.run_dir.rstrip("/\\")
    src = (a.src or run).rstrip("/\\")
    out = a.out or os.path.join(run, "trainset")
    lines_p = a.lines or os.path.join(run, "lines.csv")
    rows = list(csv.DictReader(open(lines_p, encoding="utf-8-sig")))
    if not rows:
        raise SystemExit("Empty lines.csv")
    speakers = {r['speaker'] for r in rows}
    if len(speakers) != 1:
        raise SystemExit("Export one reviewed target speaker per directory; use refine --target first")
    spk = next(iter(speakers))
    if not spk or re.search(r'[<>:"/\\|?*\x00-\x1f]', spk) or spk in ('.', '..'):
        raise SystemExit("Speaker name is not a safe output filename")
    if any(re.search(r'[|\r\n]', r['text']) for r in rows):
        raise SystemExit("Text must not contain | or line breaks")
    if not a.dry and os.path.isdir(out) and os.listdir(out):
        raise SystemExit("Output directory is not empty; choose a fresh --out")

    if a.exclude:
        ex = {(r["ep"], str(r["index"])) for r in
              csv.DictReader(open(a.exclude, encoding="utf-8-sig"))}
        # 分集的 lines.csv 没有 ep 列，用目录名兜底
        hint = (rows[0].get("ep") if rows else "") or os.path.basename(run)
        before = len(rows)
        rows = [r for r in rows
                if (r.get("ep") or hint, str(r.get("index", ""))) not in ex]
        print(f"[排除] {before} → {len(rows)} 条（本集 {hint}）")

    fld = a.file_field or ("dry_file" if rows and "dry_file" in rows[0] else "file")
    bounds = load_boundaries(src)
    print(f"词边界表：{'有 %d 个词' % len(bounds) if bounds else '缺失（无法精确判定截断）'}")

    recs = []
    for r in rows:
        p = os.path.join(run, r[fld])
        sr, x = read_wav(p)
        recs.append(dict(idx=int(r["index"]), path=p, sr=sr, x=x,
                         start=float(r["start"]), end=float(r["end"]),
                         text=r["text"], asr=float(r["asr_conf"])))

    # ---- 跨词截断修复 ----
    merged, i, nm = [], 0, 0
    while i < len(recs):
        cur = recs[i]
        while i + 1 < len(recs):
            nxt = recs[i + 1]
            gap = nxt["start"] - cur["end"]
            tail, head = cur["text"][-1:], nxt["text"][:1]
            if gap <= 0.25 and KATA.match(tail) and KATA.match(head) \
                    and len(cur["text"]) >= 4 and len(nxt["text"]) >= 3:
                cur = dict(cur, end=nxt["end"], text=cur["text"] + nxt["text"],
                           x=np.concatenate([cur["x"], nxt["x"]]),
                           idxs=cur.get("idxs", [cur["idx"]]) + [nxt["idx"]])
                nm += 1
                i += 1
                continue
            break
        merged.append(cur)
        i += 1

    keep, dropped = [], []
    for m in merged:
        dur = len(m["x"]) / m["sr"]
        nc = n_chars(m["text"])
        cps = nc / dur if dur > 0 else 0
        peak = float(np.abs(m["x"]).max()) if len(m["x"]) else 0.0
        fl = int(0.02 * m["sr"])
        if len(m["x"]) >= fl:
            fr = m["x"][: len(m["x"]) // fl * fl].reshape(-1, fl)
            e = np.sqrt((fr ** 2).mean(axis=1))
            lowq = float(np.percentile(e, 15))
        else:
            lowq = peak
        noisy = (lowq / peak) > a.max_lowq_ratio if peak > 0 else True

        rl = []          # 硬性丢弃原因
        rw = []          # 警告（保留但标记）
        if dur < a.min_dur:
            rl.append(f"过短 {dur:.2f}s")
        if dur > a.max_dur:
            rl.append(f"过长 {dur:.1f}s")
        if cps < a.min_cps:
            rl.append(f"语速低 {cps:.1f}")
        if cps > a.max_cps:
            rl.append(f"语速高 {cps:.1f}")
        if m["asr"] < a.min_asr:
            rl.append(f"ASR {m['asr']:.2f}")
        if nc < a.min_chars:
            rl.append(f"字太少 {nc}")
        if bounds:
            if cut_inside_word(m["start"], bounds):
                rl.append("句首切在词中")
            if cut_inside_word(m["end"], bounds):
                rl.append("句尾切在词中")
        if noisy and not a.keep_noisy:
            rl.append(f"底噪 {lowq/peak*100:.0f}%")
        elif noisy:
            rw.append("底噪高")
        if m["asr"] < 0.95:
            rw.append("低置信")

        if rl:
            dropped.append((m, rl, dur))
        else:
            keep.append(dict(m, dur=dur, cps=cps, nc=nc, warn=rw, noisy=noisy))

    tot = sum(k["dur"] for k in keep)
    print(f"\n输入 {len(rows)} 条，拼接修复 {nm} 处 → {len(merged)} 条")
    print(f"保留 {len(keep)} 条 / {tot:.1f}s = {tot/60:.2f} 分钟")
    print(f"丢弃 {len(dropped)} 条")
    c = Counter()
    for _, rl, _ in dropped:
        for x in rl:
            c[x.split()[0]] += 1
    print("丢弃原因:", dict(c.most_common()))
    print("\n=== 丢弃明细 ===")
    for m, rl, d in dropped:
        print(f"  #{m['idx']:3d} {d:5.2f}s {m['text'][:34]:36s} <- {'，'.join(rl)}")

    if a.dry:
        print(f"\n[dry] 保留项预览：")
        for k in keep:
            print(f"  {k['dur']:5.2f}s {k['cps']:4.1f}字/s asr{k['asr']:.2f} "
                  f"{'[' + '/'.join(k['warn']) + ']' if k['warn'] else '':12s} {k['text']}")
        return

    if not keep:
        raise SystemExit("No samples survived export filters")

    os.makedirs(out, exist_ok=True)
    spk = rows[0]["speaker"] if rows else "spk"
    lang = a.lang
    for cfgdir in (run, src):
        cfgp = os.path.join(cfgdir, "run_config.json")
        if not lang and os.path.isfile(cfgp):
            lang = json.load(open(cfgp, encoding="utf-8")).get("language") or ""
    if not lang:
        raise SystemExit("Language unavailable: set --lang explicitly or provide run_config.json")
    if not re.fullmatch(r'[a-zA-Z-]+', lang):
        raise SystemExit("Invalid language tag")

    lines = []
    for k, m in enumerate(keep, 1):
        name = f"{spk}_{k:04d}.wav"
        write_wav(os.path.join(out, name), m["x"], m["sr"])
        lines.append(f"{os.path.abspath(os.path.join(out, name))}|{spk}|{lang}|{m['text']}")
    with open(os.path.join(out, "train.list"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(out, "待校对.tsv"), "w", encoding="utf-8-sig") as f:
        f.write("序号\t时长\t字每秒\tASR\t风险\t文本\n")
        for k, m in enumerate(keep, 1):
            f.write(f"{k}\t{m['dur']:.2f}\t{m['cps']:.1f}\t{m['asr']:.2f}\t"
                    f"{'/'.join(m['warn'])}\t{m['text']}\n")

    print(f"\n已写出 {out}")
    print(f"  train.list  {len(lines)} 行 / 待校对.tsv {len(keep)} 行 / wav {len(keep)} 个")


if __name__ == "__main__":
    main()
