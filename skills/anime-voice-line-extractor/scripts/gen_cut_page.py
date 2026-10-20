#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gen_cut_page.py —— 为「损伤剔除」名单生成 A/B 复核页，方便人耳抽检。"""
import argparse
import csv
import html
import os

CODE_DESC = {
    "T1": "内容缺失",
    "T2": "文本缺字（图不对文）",
    "T3": "听不清",
    "T4": "无声",
    "T5": "时域崩塌",
}

TPL_HEAD = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>久留美 损伤剔除复核</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--dim:#6b7280;--line:#e5e7eb;--acc:#2563eb;--bad:#dc2626;--ok:#16a34a}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--card:#1c1f25;--ink:#e8eaed;--dim:#9aa3ad;--line:#2c313a;--acc:#6ea8fe;--bad:#f87171;--ok:#4ade80}}
*{box-sizing:border-box}
body{margin:0;padding:28px 20px 60px;background:var(--bg);color:var(--ink);
font:15px/1.6 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}
h1{font-size:20px;margin:0 0 6px}
.sub{color:var(--dim);font-size:13px;margin-bottom:18px;max-width:900px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:110px}
.stat b{display:block;font-size:18px}
.stat span{color:var(--dim);font-size:12px}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);
border-radius:8px;padding:12px 14px;font-size:13px;color:var(--dim);margin-bottom:20px;max-width:900px}
.note b{color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.card header{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--dim);flex-wrap:wrap}
.card header b{color:var(--ink);font-size:13px}
.tag{border-radius:4px;padding:1px 6px;font-size:11px;color:#fff;background:var(--bad)}
.dur{margin-left:auto}
.txt{margin:8px 0 4px;font-size:14px}
.txt s{color:var(--bad);text-decoration:none;opacity:.85}
.hyp{font-size:13px;color:var(--dim);margin-bottom:8px}
.metrics{display:flex;gap:6px;flex-wrap:wrap;font-size:11px;color:var(--dim);margin-bottom:8px}
.metrics i{font-style:normal;background:rgba(127,127,127,.12);border-radius:4px;padding:1px 6px}
.row{display:flex;align-items:center;gap:8px;margin-top:6px}
.row label{width:30px;font-size:12px;color:var(--dim);flex:none}
audio{width:100%;height:32px}
</style></head><body>
"""

TPL_TAIL = "</div></body></html>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = args.root
    rows = list(csv.DictReader(open(os.path.join(root, "损伤剔除.csv"), encoding="utf-8-sig")))
    out_p = args.out or os.path.join(root, "剔除复核.html")

    total_dur = sum(float(r["dur"]) for r in rows)
    cards = []
    rows.sort(key=lambda r: r["code"])

    for r in rows:
        dry_rel = r["dry"].replace("\\", "/")
        mix_rel = os.path.dirname(dry_rel).replace("roles/" + (r.get("speaker") or "久留美"), "_mix")
        # 更稳的写法：直接替换 roles/<角色> 段
        parts = dry_rel.split("/")
        if len(parts) >= 4 and parts[1] == "roles":
            mix_rel = "/".join([parts[0], "_mix", parts[-1]])
        desc = CODE_DESC.get(r["code"], r["code"])
        cards.append(f"""<div class="card">
<header><b>{html.escape(r['ep'])}#{r['index']}</b>
<span class="tag">{r['code']} {html.escape(desc)}</span>
<span class="dur">{float(r['dur']):.2f}s</span></header>
<div class="txt">原文本：{html.escape(r['text'])}</div>
<div class="hyp">干声转写：{html.escape(r['asr_hyp']) or '（空）'}</div>
<div class="metrics"><i>内容比 {float(r['len_ratio']):.2f}</i>
<i>置信 {float(r['conf']):.2f}（原 {float(r['old_conf']):.2f}）</i>
<i>字错率 {float(r['cer']):.0%}</i>
<i>包络 {float(r['env_corr'] or 0):+.2f}</i></div>
<div class="row"><label>原</label><audio controls preload="none" src="{html.escape(mix_rel)}"></audio></div>
<div class="row"><label>干</label><audio controls preload="none" src="{html.escape(dry_rel)}"></audio></div>
</div>""")

    import collections
    cnt = collections.Counter(r["code"] for r in rows)
    stat_html = "".join(
        f'<div class="stat"><b>{v}</b><span>{CODE_DESC.get(k, k)}</span></div>'
        for k, v in sorted(cnt.items()))
    stat_html = (f'<div class="stat"><b>{len(rows)}</b><span>剔除条数</span></div>'
                 f'<div class="stat"><b>{total_dur:.0f}s</b><span>剔除时长</span></div>' + stat_html)

    body = f"""<h1>损伤剔除复核（{len(rows)} 条）</h1>
<div class="sub">这些条目已从训练集中排除。每条都能直接切原声 / 干声对比。</div>
<div class="stats">{stat_html}</div>
<div class="note"><b>判据说明：</b>主判据是「文本与音频是否对得上」——
声学偏离（包络相关、谱平坦度）在 BGM 强烈的段落会系统性失效，不能单独作数。
T1 干声里少了一段内容；T2 原文本比音频少了字（图不对文，会教错模型）；
T3 置信度低且字错得多；T4 整段转写为空；T5 时域崩塌且内容也有缺失。</div>
<div class="grid">
{chr(10).join(cards)}
</div>"""

    with open(out_p, "w", encoding="utf-8") as f:
        f.write(TPL_HEAD + body + TPL_TAIL)
    print(f"→ {out_p}（{len(rows)} 条，{total_dur:.1f}s）")


if __name__ == "__main__":
    main()
