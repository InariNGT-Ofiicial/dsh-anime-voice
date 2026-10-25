#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_scn_text.py —— 从 KrKrZ 剧本的反编译产物里，把「说话人 + 台词 + 语音文件名」抽出来。

为什么这是**首选方案**，而不是 ASR：
    这个引擎的剧本里，台词原文、说话人、语音 ID、甚至官方简体中文，全都白纸黑字写着。
    ASR 只是在这些信息真的不存在时的兜底。用 ASR 去"重新听写"一份本来就有的原文，
    等于用有损手段替换无损数据 —— 而且日文 ASR 会丢标点、认错人名、写错汉字。

剧本结构（实测，不是猜的）：

    scenes[i] = {
      label: '*com_part_1:2',
      texts: [
        [
          speaker,                      # 0: 说话人（'将臣' / '運転手' / null=旁白）
          null,                         # 1: 保留位
          [
            [name, text],               # 2[0]: 日文原文   ← 与语音同语言，训练必须用这个
            [name, text],               # 2[1]: 英文
            [name, text],               # 2[2]: 简体中文   ← 官方翻译
            [name, text],               # 2[3]: 繁体中文
          ],
          [ {name, pan, type, voice} ] | null,   # 3: 语音指令，voice='uts001_001'
          duration_ms,                          # 4: 显示时长
          {data, env, ...},                     # 5: 该行的舞台/BGM 状态
        ], ...
      ],
      lines: [...],                     # 指令流；台词**不在**这里
    }

★ 关键：台词在场景级的 `texts` 里，`lines` 只放指令。
  之前按「在 lines 里找 text」去解析，只能捞到菜单项，根因就在这里。

★ 语言必须与语音一致：日文语音只能配日文原文。拿官方中文翻译去训日文语音，
  会让模型学出「中文的字、日文的音」，产出全是怪音。中文那一列只在
  「语音本身是中文」时才有用。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# 语言列索引（实测固定顺序：jp / en / cn / tw）
LANG_INDEX = {'jp': 0, 'en': 1, 'cn': 2, 'tw': 3}

KANA_RE = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')
HAN_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')

AUDIO_EXT = ('.ogg', '.wav', '.opus', '.mp3')


def strip_index_prefix(s: str) -> str:
    """「０１・芳乃」→「芳乃」"""
    t = re.sub(r'^[０-９0-9]{1,3}\s*[・･_.\-、，,．]\s*', '', s).strip()
    return t or s


def build_voice_index(voice_dir: str):
    """语音文件名（去扩展名，小写）→ 完整路径。剧本里的 voice 字段就是这个名字。"""
    idx = {}
    if not voice_dir or not os.path.isdir(voice_dir):
        return idx
    for root, _dirs, files in os.walk(voice_dir):
        for f in files:
            if not f.lower().endswith(AUDIO_EXT):
                continue
            stem = re.sub(r'\.[^.]+$', '', f).lower()
            idx.setdefault(stem, os.path.join(root, f))
    return idx


def pick_text(entry, lang: str):
    """
    从 texts[i] 里取指定语言的文本与说话人。
    语言列缺失时**回落到日文原文**，并标记 fallback —— 不静默改用别的语言。
    """
    variants = entry[2] if len(entry) > 2 and isinstance(entry[2], list) else []
    want = LANG_INDEX[lang]
    if want < len(variants) and isinstance(variants[want], list) and len(variants[want]) >= 2:
        text = variants[want][1]
        if text:
            return str(text), str(variants[want][0] or ''), False
    # 回落日文
    if variants and isinstance(variants[0], list) and len(variants[0]) >= 2 and variants[0][1]:
        return str(variants[0][1]), str(variants[0][0] or ''), True
    return '', '', False


def pick_voices(entry):
    """texts[i][3] 是语音指令数组；可能有多条（多人同时说）。返回 [voice_id, ...]"""
    v = entry[3] if len(entry) > 3 else None
    out = []
    if isinstance(v, list):
        for x in v:
            if isinstance(x, dict):
                name = x.get('voice')
                if name:
                    out.append(str(name))
    elif isinstance(v, dict):
        name = v.get('voice')
        if name:
            out.append(str(name))
    return out


def looks_like_story(doc) -> bool:
    return isinstance(doc, dict) and 'scenes' in doc and isinstance(doc.get('scenes'), list)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='从 KrKrZ 剧本 JSON 提取台词清单（lines.jsonl）')
    ap.add_argument('--scn-dir', required=True, help='含 *.ks.json 的目录')
    ap.add_argument('--out', required=True, help='输出 lines.jsonl')
    ap.add_argument('--voice-dir', default='', help='语音目录（用于校验命中率）')
    ap.add_argument('--lang', default='jp', choices=list(LANG_INDEX.keys()),
                    help='取哪一列文本；jp 才是与日文语音匹配的那一列')
    ap.add_argument('--min-chars', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args(argv)

    if not os.path.isdir(args.scn_dir):
        print(json.dumps({'ok': False, 'error': f'目录不存在: {args.scn_dir}'}, ensure_ascii=False))
        return 2

    vox = build_voice_index(args.voice_dir)
    rows = []
    stats = {
        'ok': True, 'lang': args.lang,
        'files_scanned': 0, 'files_story': 0, 'files_skipped': [],
        'entries_total': 0, 'with_text': 0, 'with_voice': 0, 'with_both': 0,
        'voice_matched': 0, 'voice_unmatched': 0,
        'fallback_to_jp': 0, 'multi_voice': 0,
        'unmatched_samples': [],
    }

    names = sorted(f for f in os.listdir(args.scn_dir) if f.endswith('.json') and not f.endswith('.resx.json'))
    for fn in names:
        path = os.path.join(args.scn_dir, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
        except Exception as e:                                # noqa: BLE001
            stats['files_skipped'].append(f'{fn}: {type(e).__name__}')
            continue
        stats['files_scanned'] += 1
        if not looks_like_story(doc):
            stats['files_skipped'].append(f'{fn}: 不是剧情文件')
            continue
        stats['files_story'] += 1

        scene_name = re.sub(r'\.json$', '', fn)
        scene_name = re.sub(r'\.ks$', '', scene_name)

        for si, sc in enumerate(doc.get('scenes') or []):
            texts = sc.get('texts') or []
            label = sc.get('label') or f's{si}'
            for ti, entry in enumerate(texts):
                if not isinstance(entry, list) or len(entry) < 3:
                    continue
                stats['entries_total'] += 1

                text, inline_name, fallback = pick_text(entry, args.lang)
                if fallback:
                    stats['fallback_to_jp'] += 1
                if not text or len(text.strip()) < args.min_chars:
                    continue
                stats['with_text'] += 1

                voices = pick_voices(entry)
                if len(voices) > 1:
                    stats['multi_voice'] += 1
                voice = voices[0] if voices else ''
                if voice:
                    stats['with_voice'] += 1

                # 说话人：texts[i][0] 优先，其次该语言列里的 name
                speaker = str(entry[0] or inline_name or '').strip()

                matched = ''
                if voice and vox:
                    matched = vox.get(voice.lower(), '')
                    if matched:
                        stats['voice_matched'] += 1
                    else:
                        stats['voice_unmatched'] += 1
                        if len(stats['unmatched_samples']) < 15:
                            stats['unmatched_samples'].append(voice)

                if text and voice:
                    stats['with_both'] += 1

                rows.append({
                    'speaker': speaker,
                    'text': text.strip(),
                    'voice': voice,
                    'voice_path': matched,
                    'voice_id': None,
                    'scene': scene_name,
                    'label': label,
                    'index': ti,
                    'src': fn,
                })
                if args.limit and len(rows) >= args.limit:
                    break
            if args.limit and len(rows) >= args.limit:
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 说话人分布（只统计有语音的，旁白没语音没意义）
    spk = {}
    for r in rows:
        if r['voice']:
            key = strip_index_prefix(r['speaker']) if r['speaker'] else '(未标注)'
            spk[key] = spk.get(key, 0) + 1
    stats['speakers'] = sorted(spk.items(), key=lambda kv: -kv[1])[:40]

    stats['rows'] = len(rows)
    stats['out'] = os.path.abspath(args.out)
    if not rows:
        stats['ok'] = False
        stats['error'] = '一条都没抽到：确认 --scn-dir 指向反编译出的 *.ks.json（不是 .scn 二进制）'

    print(json.dumps(stats, ensure_ascii=False))
    return 0 if rows else 3


if __name__ == '__main__':
    sys.exit(main())
