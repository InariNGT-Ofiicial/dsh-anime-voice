# -*- coding: utf-8 -*-
"""把 refine 产物渲染成单文件 HTML 预览页：全部条目、可逐条试听、带筛选。"""
import csv
import html
import json
import os
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
OUT = os.path.join(RUN, "预览.html")

rows = list(csv.DictReader(open(os.path.join(RUN, "lines.csv"), encoding="utf-8-sig")))
cfg = {}
p = os.path.join(RUN, "run_config.json")
if os.path.isfile(p):
    try:
        cfg = json.load(open(p, encoding="utf-8"))
    except Exception:
        cfg = {}

total = sum(float(r["dur"]) for r in rows)
spk = sorted({r["speaker"] for r in rows})
src = os.path.basename(cfg.get("input") or "")

def mmss(t):
    t = float(t)
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"

items = []
for r in rows:
    f = r["file"].replace("\\", "/")
    items.append(f"""<tr data-t="{html.escape(r['text'])}">
  <td class="n">{r['index']}</td>
  <td class="t">{mmss(r['start'])}</td>
  <td class="d">{float(r['dur']):.1f}s</td>
  <td class="x">{html.escape(r['text'])}</td>
  <td class="a"><audio controls preload="none" src="{f}"></audio></td>
</tr>""")

body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>久留美台词切片 · 试听</title>
<style>
  :root {{ --fg:#1a1a1a; --dim:#6b7280; --line:#e5e7eb; --bg:#fbfbfc; --card:#fff; --accent:#b91c1c; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e8e8ea; --dim:#9aa0a6; --line:#2c2f36; --bg:#15171a; --card:#1c1f24; --accent:#f87171; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:22px; margin:0 0 6px; letter-spacing:.5px; }}
  .meta {{ color:var(--dim); font-size:13px; margin-bottom:22px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:20px; }}
  .stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
           padding:10px 16px; min-width:96px; }}
  .stat b {{ display:block; font-size:20px; line-height:1.2; }}
  .stat span {{ color:var(--dim); font-size:12px; }}
  .bar {{ display:flex; gap:10px; margin-bottom:14px; align-items:center; }}
  input[type=search] {{ flex:1; padding:9px 12px; border:1px solid var(--line);
    border-radius:9px; background:var(--card); color:var(--fg); font-size:14px; }}
  input[type=search]:focus {{ outline:2px solid var(--accent); outline-offset:-1px; }}
  .hint {{ color:var(--dim); font-size:12px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card);
           border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
  thead th {{ text-align:left; font-size:12px; color:var(--dim); font-weight:600;
              padding:10px 12px; border-bottom:1px solid var(--line); background:var(--bg); }}
  td {{ padding:8px 12px; border-bottom:1px solid var(--line); vertical-align:middle; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:color-mix(in srgb, var(--accent) 4%, transparent); }}
  td.n {{ color:var(--dim); font-variant-numeric:tabular-nums; width:44px; }}
  td.t, td.d {{ white-space:nowrap; color:var(--dim); font-variant-numeric:tabular-nums;
                font-size:13px; }}
  td.d {{ width:62px; }}
  td.x {{ font-size:15px; }}
  td.a {{ width:230px; }}
  audio {{ width:220px; height:32px; }}
  mark {{ background:var(--accent); color:#fff; border-radius:3px; padding:0 2px; }}
  footer {{ margin-top:26px; color:var(--dim); font-size:12px; line-height:1.9; }}
  code {{ background:var(--card); border:1px solid var(--line); border-radius:4px;
          padding:1px 5px; font-size:12px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>久留美 · 台词切片</h1>
  <div class="meta">{html.escape(src)} · 已按「严苛清洗」剔除混入他人对话的切片</div>

  <div class="stats">
    <div class="stat"><b>{len(rows)}</b><span>条台词</span></div>
    <div class="stat"><b>{total:.0f}s</b><span>总时长</span></div>
    <div class="stat"><b>{total/max(len(rows),1):.1f}s</b><span>平均句长</span></div>
    <div class="stat"><b>{len(spk)}</b><span>角色</span></div>
  </div>

  <div class="bar">
    <input type="search" id="q" placeholder="搜索台词（支持日文）…">
    <span class="hint" id="cnt"></span>
  </div>

  <table>
    <thead><tr><th>#</th><th>时间</th><th>时长</th><th>台词</th><th>试听</th></tr></thead>
    <tbody id="tb">
{os.linesep.join(items)}
    </tbody>
  </table>

  <footer>
    产物：<code>roles/{html.escape(spk[0])}/</code> 每条 wav + 同名 txt ·
    索引 <code>lines.csv</code> · 训练清单 <code>gpt_sovits.list</code>（GPT-SoVITS 可直接用）<br>
    已知局限：重叠说话（两人同时发声）无法靠声纹分离，这类切片已整条丢弃；
    个别句首若发音极轻仍可能缺字。
  </footer>
</div>
<script>
const q = document.getElementById('q'), tb = document.getElementById('tb');
const rows = [...tb.querySelectorAll('tr')], cnt = document.getElementById('cnt');
function upd(){{
  const v = q.value.trim().toLowerCase();
  let n = 0;
  rows.forEach(r => {{
    const hit = !v || r.dataset.t.toLowerCase().includes(v);
    r.style.display = hit ? '' : 'none';
    if (hit) n++;
  }});
  cnt.textContent = n + ' / ' + rows.length;
}}
q.addEventListener('input', upd); upd();
// 只允许一个音频同时播放
document.addEventListener('play', e => {{
  if (e.target.tagName !== 'AUDIO') return;
  document.querySelectorAll('audio').forEach(a => {{ if (a !== e.target) a.pause(); }});
}}, true);
</script>
</body>
</html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(body)
print("OK", OUT, f"{len(rows)} 条")
