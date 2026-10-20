#!/usr/bin/env python
"""Synthesize one utterance with explicitly selected local GPT-SoVITS v2 weights."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def preserve_weight_selection(root):
    """Upstream persists its last selection even during library inference."""
    path=root/'weight.json'
    original=path.read_bytes() if path.exists() else None
    try:
        yield
    finally:
        if original is None:
            if path.exists(): path.unlink()
        elif path.read_bytes() != original:
            path.write_bytes(original)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=os.environ.get('GSV_ROOT'), type=Path)
    ap.add_argument('--sovits', required=True, type=Path)
    ap.add_argument('--gpt', required=True, type=Path)
    ap.add_argument('--ref', required=True, type=Path)
    ap.add_argument('--prompt-text', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--lang', choices=['ja', 'zh', 'en'], default='ja')
    ap.add_argument('--prompt-lang', choices=['ja', 'zh', 'en'], default='ja')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = ap.parse_args()
    if not a.root: ap.error('--root/GSV_ROOT is required')
    a.root = a.root.resolve()
    for field in ['sovits', 'gpt', 'ref']:
        p = getattr(a, field).resolve()
        if not p.is_file(): ap.error(f'missing {field}: {p}')
        setattr(a, field, p)
    out = a.out.resolve()
    if out.exists() or out.with_suffix('.json').exists(): ap.error('output exists; choose a fresh --out')
    import soundfile as sf
    duration = sf.info(a.ref).duration
    if not 3 <= duration <= 10: ap.error('reference must be 3–10 seconds')
    os.environ.update(version='v2', gpt_path=str(a.gpt), sovits_path=str(a.sovits))
    if a.device == 'cpu':
        os.environ.update(CUDA_VISIBLE_DEVICES='', is_half='False')
    sys.path.insert(0, str(a.root))
    os.chdir(a.root)
    import random
    import numpy as np
    import torch
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    with preserve_weight_selection(a.root):
        from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
        from tools.i18n.i18n import I18nAuto
        i18n = I18nAuto()
        change_gpt_weights(gpt_path=str(a.gpt))
        # Some upstream versions implement this as a generator; exhaust it to load weights.
        changed = change_sovits_weights(sovits_path=str(a.sovits))
        if hasattr(changed, '__next__'): list(changed)
        langs = {'ja': '日文', 'zh': '中文', 'en': '英文'}
        result = list(get_tts_wav(ref_wav_path=str(a.ref), prompt_text=a.prompt_text,
            prompt_language=i18n(langs[a.prompt_lang]), text=a.text,
            text_language=i18n(langs[a.lang]), top_p=1, temperature=1))
        if not result: raise SystemExit('synthesizer returned no audio')
        sr, audio = result[-1]
        if len(audio) == 0 or not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
            raise SystemExit('invalid/silent output')
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, audio, sr)
    metadata = dict(text=a.text, lang=a.lang, sample_rate=sr, samples=len(audio),
        duration=len(audio)/sr, seed=a.seed, output_sha256=sha256(out),
        gpt_sha256=sha256(a.gpt), sovits_sha256=sha256(a.sovits),
        reference_sha256=sha256(a.ref), reference_duration=duration, upstream_weight_selection_restored=True,
        torch=torch.__version__, cuda=torch.version.cuda, device='cuda' if torch.cuda.is_available() else 'cpu',
        claim='technical synthesis smoke test; listening and transcription still required')
    out.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
