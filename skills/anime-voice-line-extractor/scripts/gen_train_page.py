#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_train_page.py —— 生成训练集校对页（按风险排序 + 可直接改文本 + 一键导出 train.list）。

同音错字只能人耳听出来，但 157 条全听一遍不现实。按「干声重转写 vs 原文本」
的字错率分级后，高/中两档通常只占四分之一，先看这些就够。
"""
import argparse
import csv
import html
import os

HEAD = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>久留美 训练集校对</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--dim:#6b7280;--line:#e5e7eb;--acc:#2563eb;
--hi:#dc2626;--md:#d97706;--lo:#16a34a}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--card:#1c1f25;--ink:#e8eaed;--dim:#9aa3ad;
--line:#2c313a;--acc:#6ea8fe;--hi:#f87171;--md:#fbbf24;--lo:#4ade80}}
*{box-sizing:border-box}
body{margin:0;padding:24px 18px 90px;background:var(--bg);color:var(--ink);
font:15px/1.6 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}
h1{font-size:20px;margin:0 0 6px}
.sub{color:var(--dim);font-size:13px;margin-bottom:16px;max-width:880px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:9px 14px;min-width:96px}
.stat b{display:block;font-size:18px}
.stat span{color:var(--dim);font-size:12px}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);padding:8px 0 10px;
border-bottom:1px solid var(--line);margin-bottom:14px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.bar button{background:var(--card);border:1px solid var(--line);color:var(--ink);
border-radius:8px;padding:6px 12px;font-size:13px;cursor:pointer}
.bar button.on{border-color:var(--acc);color:var(--acc)}
.bar .grow{flex:1}
.bar .export{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(370px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:11px 13px}
.card.hi{border-left:3px solid var(--hi)}
.card.md{border-left:3px solid var(--md)}
.card.lo{border-left:3px solid var(--lo)}
header{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--dim);flex-wrap:wrap}
header b{color:var(--ink);font-size:13px}
.tag{border-radius:4px;padding:1px 6px;font-size:11px;color:#fff}
.hi .tag{background:var(--hi)}.md .tag{background:var(--md)}.lo .tag{background:var(--lo)}
.unk .tag{background:#6b7280}
.dur{margin-left:auto}
.tx{margin:7px 0 4px;width:100%;background:transparent;border:1px solid transparent;
border-radius:6px;padding:4px 6px;color:var(--ink);font:inherit;font-size:14.5px;resize:none;min-height:2.4em}
.tx:hover{border-color:var(--line)}
.tx:focus{outline:none;border-color:var(--acc);background:rgba(127,127,127,.06)}
.hyp{font-size:12.5px;color:var(--dim);margin-bottom:6px;word-break:break-all}
audio{width:100%;height:30px;margin-top:2px}
</style></head><body>
"""

BODY_HEAD = """<h1>训练集校对（{n} 条 / {mins:.1f} 分钟）</h1>
<div class="sub">按「干声重转写 vs 原文本」的字错率排序。<b>高</b> 是转写与原文本差异大（多半原文本缺字或被截断），
<b>中</b> 有同音替换疑点，<b>低</b> 基本可信。文本框可直接改，改完点右上「导出 train.list」下载。</div>
<div class="stats">{stats}</div>
<div class="bar">
<button data-f="hi" class="on">高 {nh}</button>
<button data-f="md" class="on">中 {nm}</button>
<button data-f="lo">低 {nl}</button>
<button data-f="all">全部</button>
<span class="grow"></span>
<span id="cnt" class="sub" style="margin:0"></span>
<button class="export" onclick="dl()">导出 train.list</button>
</div>
<div class="grid" id="grid">
"""

TAIL = """</div>
<script>
const boxes=[...document.querySelectorAll('.card')];
function apply(f){
  boxes.forEach(b=>{
    b.style.display = (f==='all'||b.dataset.lv===f)?'':'none';
  });
  document.querySelectorAll('.bar button[data-f]').forEach(x=>
    x.classList.toggle('on', x.dataset.f===f));
  count();
}
function count(){
  document.getElementById('cnt').textContent =
    '显示 '+boxes.filter(b=>b.style.display!=='none').length+' 条';
}
document.querySelectorAll('.bar button[data-f]').forEach(btn=>{
  btn.onclick=()=>apply(btn.dataset.f);
});
function dl(){
  const rows=[['路径','角色','语言','文本']];
  boxes.forEach(b=>{
    const t=b.querySelector('.tx').value.trim();
    rows.push([b.dataset.path, '久留美', 'ja', t]);
  });
  const lines=rows.map(r=>r.join('|')).join('\\n');
  const blob=new Blob(['\\ufeff'+lines],{type:'text/plain;charset=utf-8'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='train.list';
  a.click();
}
// 默认只显示高 + 中两档
boxes.forEach(b=>{ b.style.display = (b.dataset.lv==='低')?'none':''; });
count();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainset", required=True)
    args = ap.parse_args()
    ts = args.trainset

    rows = list(csv.DictReader(open(os.path.join(ts, "待校对_分级.tsv"), encoding="utf-8-sig"),
                               delimiter="\t"))
    base = os.path.abspath(ts)
    cards = []
    n = {"高": 0, "中": 0, "低": 0, "未知": 0}
    total = 0.0
    for r in rows:
        lvl = r["风险级别"] or "未知"
        n[lvl] = n.get(lvl, 0) + 1
        total += float(r["时长"] or 0)
        rel = os.path.relpath(r["文件"], base).replace("\\", "/") if r["文件"] else ""
        cls = {"高": "hi", "中": "md", "低": "lo"}.get(lvl, "unk")
        hyp = r["干声转写"] or ""
        hyp_html = ""
        if hyp and hyp != r["文本"]:
            hyp_html = f'<div class="hyp">干声转写：{html.escape(hyp)}　字错率 {r["字错率"]}</div>'
        elif hyp:
            hyp_html = f'<div class="hyp">干声转写一致</div>'
        cards.append(f"""<div class="card {cls}" data-lv="{lvl}" data-path="{html.escape(r['文件'])}">
<header><b>{html.escape(r['集'])}#{r['序号']}</b>
<span class="tag">{lvl}</span>
<span>{r['时长']}s</span>
<span class="dur">ASR {r['ASR']}</span></header>
<textarea class="tx" rows="1">{html.escape(r['文本'])}</textarea>
{hyp_html}
<audio controls preload="none" src="{html.escape(rel)}"></audio>
</div>""")

    stats = (f'<div class="stat"><b>{len(rows)}</b><span>条数</span></div>'
             f'<div class="stat"><b>{total:.0f}s</b><span>总时长</span></div>'
             f'<div class="stat"><b style="color:var(--hi)">{n["高"]}</b><span>高风险</span></div>'
             f'<div class="stat"><b style="color:var(--md)">{n["中"]}</b><span>待疑</span></div>'
             f'<div class="stat"><b style="color:var(--lo)">{n["低"]}</b><span>可信</span></div>')

    out = os.path.join(ts, "训练集校对.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(HEAD)
        f.write(BODY_HEAD.format(n=len(rows), mins=total / 60, stats=stats,
                                 nh=n["高"], nm=n["中"], nl=n["低"]))
        f.write("\n".join(cards))
        f.write(TAIL)
    print(f"→ {out}（{len(rows)} 条，高 {n['高']} / 中 {n['中']} / 低 {n['低']} / 未知 {n['未知']}）")


if __name__ == "__main__":
    main()
