# -*- coding: utf-8 -*-
"""make_dataset_report.py —— 训练集重建前后对照页（S5 抽检用，新主路线）。

回答一个问题：新数据集里「一句话说完」的比例到底改变了多少。
指标全部可复算：句末标点数、以助词/词干收尾的比例、时长分布、覆盖时长。
同时随机抽 24 条嵌进页面，直接听句尾（自然收住 / 说到一半被掐）。

用法：
  $PY make_dataset_report.py --new _full/out/ts/train.list --out out/对照.html
  $PY make_dataset_report.py --new … --old 旧路线/train.list   # 有旧集才出双列对照
"""
import argparse
import io
import os
import random

import numpy as np
import soundfile as sf

DEFAULT_BASE = os.environ.get("ANIME_WORK", os.getcwd())
ENDERS = "。！？!?…"
PARTS = ("を", "は", "が", "に", "で", "と", "へ", "も", "の", "や", "から", "まで",
         "けど", "って", "て", "し", "ば", "たら", "なら")


def read_list(p):
    rows = []
    for l in io.open(p, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        f, spk, lang, text = l.split("|", 3)
        d = sf.info(f).duration if os.path.isfile(f) else 0.0
        rows.append(dict(path=f, text=text.strip(), dur=d))
    return rows


def stats(rows, name):
    n = len(rows)
    if not n:
        raise SystemExit("Empty dataset")
    dur = np.array([r["dur"] for r in rows])
    end = sum(1 for r in rows if r["text"].rstrip().endswith(tuple(ENDERS)))
    part = sum(1 for r in rows if r["text"].rstrip().endswith(PARTS))
    return dict(name=name, n=n, total=float(dur.sum()), med=float(np.median(dur)),
                ended=end, ended_pct=end / n * 100, part=part, part_pct=part / n * 100)


def main():
    ap = argparse.ArgumentParser(description="训练集重建前后对照页")
    ap.add_argument("--new", default=os.path.join(DEFAULT_BASE, "_full/out/ts/train.list"),
                    help="新清单（实际喂进训练的那个）")
    ap.add_argument("--old", default="", help="旧路线 train.list，给了才出双列对照")
    ap.add_argument("--out", default=os.path.join(DEFAULT_BASE, "out", "训练集重建对照.html"))
    a = ap.parse_args()

    new = read_list(a.new)
    old = read_list(a.old) if a.old and os.path.isfile(a.old) else []
    sn = stats(new, "新（整集正片按标点切句）")
    so = stats(old, "旧（pack 切句）") if old else None

    print(f"新 {sn}")
    if so:
        print(f"旧 {so}")

    random.seed(11)
    samples = random.sample(new, min(24, len(new)))

    rows_html = []
    out_html = a.out
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    for r in samples:
        nm = os.path.basename(r["path"])
        try:
            rel = os.path.relpath(r["path"], os.path.dirname(os.path.abspath(out_html)))
            rel = rel.replace("\\", "/")
        except ValueError:      # 页面和音频不在同一个盘，只能给绝对 file:// 地址
            rel = "file:///" + r["path"].replace("\\", "/")
        ended = r["text"].rstrip().endswith(tuple(ENDERS))
        tag = ('<span class="ok">有句末标点（需试听）</span>' if ended
               else '<span class="warn">无句末标点（需试听）</span>')
        rows_html.append(f"""
        <div class="row">
          <div class="meta"><b>{nm}</b> · {r['dur']:.2f}s · {tag}</div>
          <div class="txt">{r['text']}</div>
          <audio controls preload="none" src="{rel}"></audio>
        </div>""")

    def bar(v, mx, color):
        return (f'<div class="bar"><i style="width:{min(100, v/mx*100):.1f}%;'
                f'background:{color}"></i><b>{v:.1f}%</b></div>')

    if so:
        core = f"""<table>
<tr><th>指标</th><th>旧（pack 切句）</th><th>新（整集正片）</th></tr>
<tr><td>样本数</td><td class="num">{so['n']}</td><td class="num">{sn['n']}</td></tr>
<tr><td>总时长</td><td class="num">{so['total']/60:.1f} 分钟</td><td class="num">{sn['total']/60:.1f} 分钟</td></tr>
<tr><td>单条中位时长</td><td class="num">{so['med']:.2f}s</td><td class="num">{sn['med']:.2f}s</td></tr>
<tr><td><b>句末有句号/问号/叹号</b></td><td>{bar(so['ended_pct'],100,'#ff6b6b')}</td><td>{bar(sn['ended_pct'],100,'#3ddc84')}</td></tr>
<tr><td>以助词收尾（语法上必须还有下文）</td><td>{bar(so['part_pct'],60,'#ff6b6b')}</td><td>{bar(sn['part_pct'],60,'#3ddc84')}</td></tr>
</table>
<div style="color:var(--dim);font-size:13px">旧集 <code>97.5%</code> 的样本没有句末标点 ——
这不是标点问题，是这些句子本来就切在话说一半的地方。</div>"""
    else:
        core = f"""<table>
<tr><th>指标</th><th>本批（整集正片）</th></tr>
<tr><td>样本数</td><td class="num">{sn['n']}</td></tr>
<tr><td>总时长</td><td class="num">{sn['total']/60:.1f} 分钟</td></tr>
<tr><td>单条中位时长</td><td class="num">{sn['med']:.2f}s</td></tr>
<tr><td><b>句末有句号/问号/叹号</b></td><td>{bar(sn['ended_pct'],100,'#3ddc84')}</td></tr>
<tr><td>以助词收尾（语法上必须还有下文）</td><td>{bar(sn['part_pct'],60,'#3ddc84')}</td></tr>
</table>
<div style="color:var(--dim);font-size:13px">闸门：句末标点 ≥90%、助词收尾 ≤5%。</div>"""

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>训练集重建对照</title><style>
:root{{--bg:#0f1115;--card:#181b22;--fg:#e8eaed;--dim:#9aa0a6;--ok:#3ddc84;--warn:#ffb300;--accent:#7aa2f7}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;padding:32px 24px 80px}}
.wrap{{max-width:980px;margin:0 auto}}
h1{{font-size:24px;margin:0 0 6px}}
h2{{font-size:17px;margin:34px 0 10px;padding-left:10px;border-left:3px solid var(--accent)}}
.sub{{color:var(--dim);font-size:13.5px;margin-bottom:22px}}
table{{width:100%;border-collapse:collapse;margin:10px 0 6px;font-size:14px}}
th,td{{padding:9px 10px;border-bottom:1px solid #262a33;text-align:left}}
th{{color:var(--dim);font-weight:500;font-size:13px}}
td.num{{font-variant-numeric:tabular-nums}}
.bar{{position:relative;background:#22262e;border-radius:4px;height:20px;min-width:170px}}
.bar i{{position:absolute;left:0;top:0;bottom:0;border-radius:4px;opacity:.55}}
.bar b{{position:relative;padding-left:8px;font-size:12.5px;line-height:20px}}
.card{{background:var(--card);border:1px solid #232730;border-radius:10px;padding:16px 18px;margin:12px 0}}
.row{{padding:11px 0;border-bottom:1px solid #232730}}
.row:last-child{{border-bottom:none}}
.meta{{color:var(--dim);font-size:12.5px;margin-bottom:4px}}
.txt{{margin-bottom:7px;font-size:14.5px}}
audio{{width:100%;height:32px}}
.ok{{color:var(--ok)}} .warn{{color:var(--warn)}}
code{{background:#22262e;padding:1px 5px;border-radius:4px;font-size:13px}}
.big{{font-size:30px;font-weight:600;color:var(--ok)}}
</style></head><body><div class="wrap">
<h1>训练集：从「半句」到「整句」</h1>
<div class="sub">新集来自整集正片（只按句末标点断），旧集来自 5 分钟 pack 切句（按停顿断）。
下面每个数字都能用 <code>split_full.py --plan</code> 复算。</div>

<h2>核心差别</h2>
<div class="card">
{core}
</div>

<h2>抽样试听（新集）</h2>
<div class="sub">随机 24 条。听句尾：是「说完收住」，还是「说到一半被掐」。</div>
<div class="card">{''.join(rows_html)}</div>

<h2>怎么复现</h2>
<div class="card" style="font-size:13.5px;color:var(--dim)">
<pre style="margin:0;white-space:pre-wrap">$PY asr_full2.py --base $W      # 整集正片 → 分块 ASR（每块注入带标点提示词）
$PY split_full.py --slice       # 只按句末标点切句，从连续干声取自然衰减
$PY filter_speaker.py --ref …   # 用已确认同人样本的声纹质心 + F0 闸门筛句</pre></div>
</div></body></html>"""

    io.open(out_html, "w", encoding="utf-8").write(html)
    print("已写", out_html)


if __name__ == "__main__":
    main()
