# -*- coding: utf-8 -*-
"""把多集的 refine 产物合并成一个可逐条试听的 HTML 预览页。

用法：
    python gen_preview_multi.py out/久留美_三集预览.html \
        EP01=out/FX久留美_台词_最终 EP02=out/久留美_EP02_v2 EP03=out/久留美_EP03_v3
"""
import csv
import html
import json
import os
import sys
from datetime import timedelta

OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "预览.html"
BASE = os.path.dirname(OUT)
SPECS = []
for a in sys.argv[2:]:
    if "=" not in a:
        continue
    name, path = a.split("=", 1)
    SPECS.append((name.strip(), os.path.abspath(path.strip())))
if not SPECS:
    print("用法: gen_preview_multi.py <out.html> EP01=<dir> EP02=<dir> ...")
    raise SystemExit(2)


def mmss(t):
    t = float(t)
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def esc(s):
    return html.escape(s or "")


groups = []
grand_n, grand_d = 0, 0.0
for name, run in SPECS:
    p = os.path.join(run, "lines.csv")
    if not os.path.isfile(p):
        print(f"[skip] {name}: 缺 lines.csv ({p})")
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    for r in rows:
        r["_src"] = os.path.relpath(os.path.join(run, r["file"]), BASE).replace("\\", "/")
    d = sum(float(r["dur"]) for r in rows)
    grand_n += len(rows)
    grand_d += d
    groups.append((name, rows, d))

parts = []
for name, rows, d in groups:
    cards = []
    for i, r in enumerate(rows, 1):
        warn = []
        if float(r["dur"]) < 1.2:
            warn.append("过短")
        if float(r.get("asr_conf") or 1) < 0.85:
            warn.append("低置信")
        tag = ('<span class="w">' + "/".join(warn) + "</span>") if warn else ""
        cards.append(f"""    <li class="row" data-q="{esc(r['text'])}">
      <span class="idx">{i}</span>
      <span class="t">{mmss(r['start'])}</span>
      <span class="d">{float(r['dur']):.2f}s</span>
      <button class="play" data-src="{r['_src']}">▶</button>
      <span class="txt">{esc(r['text'])}{tag}</span>
    </li>""")
    parts.append(f"""<section class="grp">
  <h2>{esc(name)} <em>{len(rows)} 条 · {d:.1f}s · {timedelta(seconds=int(d))}</em></h2>
  <ol class="list">
{chr(10).join(cards)}
  </ol>
</section>""")

doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>久留美台词试听 · {grand_n} 条</title>
<style>
  :root{{--bg:#f7f8fa;--card:#fff;--line:#e6e8ee;--ink:#1c2230;--dim:#6b7688;--accent:#c2456a;}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.6 "Microsoft YaHei","PingFang SC",system-ui,sans-serif}}
  header{{position:sticky;top:0;z-index:9;background:rgba(247,248,250,.94);
    backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:14px 22px}}
  h1{{margin:0 0 4px;font-size:19px;letter-spacing:.02em}}
  .sum{{color:var(--dim);font-size:13px}}
  .sum b{{color:var(--accent);font-size:15px}}
  input{{margin-top:10px;width:100%;max-width:340px;padding:8px 12px;border:1px solid var(--line);
    border-radius:8px;font-size:14px;outline:none;background:#fff}}
  input:focus{{border-color:var(--accent)}}
  main{{padding:18px 22px 60px;max-width:1080px}}
  .grp{{margin-bottom:26px}}
  h2{{font-size:15px;margin:0 0 10px;padding-left:9px;border-left:3px solid var(--accent)}}
  h2 em{{font-style:normal;color:var(--dim);font-weight:400;font-size:12.5px;margin-left:6px}}
  .list{{list-style:none;margin:0;padding:0;background:var(--card);border:1px solid var(--line);
    border-radius:10px;overflow:hidden}}
  .row{{display:flex;align-items:center;gap:10px;padding:8px 12px;border-top:1px solid var(--line);
    font-size:14px}}
  .row:first-child{{border-top:none}}
  .row:hover{{background:#fbfbfd}}
  .idx{{width:30px;color:var(--dim);font-size:12px;text-align:right;flex:none}}
  .t{{width:46px;color:var(--dim);font-size:12.5px;font-variant-numeric:tabular-nums;flex:none}}
  .d{{width:44px;color:#9aa4b5;font-size:12px;font-variant-numeric:tabular-nums;flex:none}}
  .play{{flex:none;width:28px;height:28px;border-radius:50%;border:1px solid var(--line);
    background:#fff;color:var(--accent);cursor:pointer;font-size:11px;line-height:1}}
  .play:hover{{background:var(--accent);color:#fff;border-color:var(--accent)}}
  .play.on{{background:var(--accent);color:#fff}}
  .txt{{flex:1;word-break:break-all}}
  .w{{margin-left:6px;padding:1px 6px;border-radius:4px;background:#fff2f5;color:var(--accent);
    font-size:11px;white-space:nowrap}}
  .hide{{display:none}}
  audio{{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);width:min(560px,92vw);
    border-radius:10px;box-shadow:0 6px 24px rgba(20,30,50,.14);display:none;background:#fff}}
  audio.show{{display:block}}
</style></head><body>
<header>
  <h1>久留美 台词切片 · 试听</h1>
  <div class="sum">共 <b>{grand_n}</b> 条 · <b>{grand_d:.1f}s</b>（{timedelta(seconds=int(grand_d))}）·
    {len(groups)} 集 · 已剔除混入的他人对话与 OP/ED</div>
  <input id="q" placeholder="搜索台词…（输入过滤）">
</header>
<main>
{chr(10).join(parts)}
</main>
<audio id="ap" controls></audio>
<script>
const ap=document.getElementById('ap');
document.querySelectorAll('.play').forEach(b=>{{
  b.addEventListener('click',()=>{{
    const playing = b.classList.contains('on');
    document.querySelectorAll('.play.on').forEach(x=>x.classList.remove('on'));
    if(playing){{ap.pause();ap.classList.remove('show');return;}}
    ap.src=b.dataset.src; ap.classList.add('show');
    ap.play().catch(e=>alert('无法播放：'+e.message));
    b.classList.add('on');
  }});
}});
ap.addEventListener('ended',()=>{{document.querySelectorAll('.play.on').forEach(x=>x.classList.remove('on'));}});
document.getElementById('q').addEventListener('input',e=>{{
  const k=e.target.value.trim();
  document.querySelectorAll('.row').forEach(r=>{{
    r.classList.toggle('hide', k && !r.dataset.q.includes(k));
  }});
}});
</script>
</body></html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"已写出 {OUT}")
print(f"  {grand_n} 条 / {grand_d:.1f}s / {len(groups)} 集")
