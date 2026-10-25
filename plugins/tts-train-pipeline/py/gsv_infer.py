#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gsv_infer.py —— 用**训练出来的**权重合成一句话，产出可直接听的 wav。

关键点（每一条都踩过）：
    1. 必须在 gsvDir 里 chdir：tts_infer.yaml 的路径、bert/hubert 的相对路径都是相对工程根的。
    2. version 必须与权重版本一致 —— v2ProPlus 的 .pth 用 v2 的 config 加载会直接崩，
       或（更糟）不崩但音质全错。
    3. 走临时 yaml 而不是 init_*_weights()：TTS_Config 读 custom 段时就带上了权重路径，
       构造即加载，少一次"到底加载没加载"的歧义。产物自证靠 vram 与合成结果。
    4. 代理变量必须清掉：本机实测带系统代理时，本地 model 加载会莫名其妙地慢或卡。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import wave


def load_torch_pipeline(gsv_dir: str, version: str, sovits: str, gpt: str,
                        device: str, is_half: bool, tmp_dir: str):
    os.chdir(gsv_dir)
    sys.path.insert(0, gsv_dir)
    sys.path.insert(0, os.path.join(gsv_dir, "GPT_SoVITS"))
    os.environ["version"] = version
    os.environ["PATH"] = gsv_dir + os.pathsep + os.environ.get("PATH", "")
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    os.environ.setdefault("HF_HOME", os.path.join(gsv_dir, "_hf_home"))

    import yaml

    base_yaml = os.path.join(gsv_dir, "GPT_SoVITS/configs/tts_infer.yaml")
    with open(base_yaml, "r", encoding="utf-8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader) or {}

    custom = dict(conf.get("custom") or {})
    custom.update({
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "device": device,
        "is_half": bool(is_half),
        "t2s_weights_path": gpt,
        "vits_weights_path": sovits,
        "version": version,
    })
    conf["custom"] = custom

    os.makedirs(tmp_dir, exist_ok=True)
    cfg_path = os.path.join(tmp_dir, f"infer_{os.getpid()}.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(conf, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    from TTS_infer_pack.TTS import TTS, TTS_Config

    cfg = TTS_Config(cfg_path)
    if getattr(cfg, "version", None) != version:
        raise RuntimeError(f"config 版本不符：期望 {version}，实际 {cfg.version}")
    tts = TTS(cfg)
    return tts, cfg


def guess_lang(text: str, fallback: str = "zh") -> str:
    """按字符构成判语言。假名必须先判（日文里汉字和假名混排）。"""
    if not text:
        return fallback
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return fallback


def trim_tail(x, sr: int, rel_db: float = -40.0, floor_sec: float = 0.02):
    """裁掉尾部静音，但保留 floor_sec 的余量。

    ★ 为什么必须裁：cut5 会把长文本切成多句分别合成，每段音频尾部都自带
      约 0.9 秒的静音（模型从参考音频那里学来的「说完留白」）。上游再在段间
      插入 fragment_interval（默认 0.3s），两者叠加 ⇒ 实测每个句间停顿长达
      **1.2 秒**，12 秒的文本里有 5.3 秒是静音。听感就是逐句往外蹦，
      再叠加每句独立的语调包络，就成了「外国人念汉文」。
    """
    import numpy as np
    if x.size == 0:
        return x
    peak = float(np.abs(x).max())
    if peak <= 0:
        return x
    thr = peak * (10 ** (rel_db / 20.0))
    loud = np.flatnonzero(np.abs(x) > thr)
    if loud.size == 0:
        return x[: int(sr * floor_sec)]
    end = min(x.size, loud[-1] + int(sr * floor_sec))
    return x[:end]


def trim_head(x, sr: int, rel_db: float = -40.0, floor_sec: float = 0.01):
    """裁掉首部静音。段首留白同样会让字与字之间发空。"""
    import numpy as np
    if x.size == 0:
        return x
    peak = float(np.abs(x).max())
    if peak <= 0:
        return x
    thr = peak * (10 ** (rel_db / 20.0))
    loud = np.flatnonzero(np.abs(x) > thr)
    if loud.size == 0:
        return x[: int(sr * floor_sec)]
    start = max(0, loud[0] - int(sr * floor_sec))
    return x[start:]


def join_fragments(frags, sr: int, gap_sec: float = 0.22):
    """把分段音频裁剪首尾后按 gap_sec 拼接。"""
    import numpy as np
    usable = [f for f in frags if f is not None and f.size > 0]
    if not usable:
        return None
    gap = np.zeros(int(sr * gap_sec), dtype=usable[0].dtype)
    out = []
    for i, f in enumerate(usable):
        f = trim_head(f, sr)
        f = trim_tail(f, sr)
        if f.size == 0:
            continue
        if out:
            out.append(gap)
        out.append(f)
    return np.concatenate(out) if out else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="用训好的 GPT-SoVITS 权重合成一句")
    ap.add_argument("--gsv-dir", required=True)
    ap.add_argument("--version", default="v2ProPlus")
    ap.add_argument("--sovits", required=True)
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--ref", required=True, help="参考音频（决定音色底子与情感基调）")
    ap.add_argument("--ref-text", default="", help="参考音频的台词；留空则由模型自行推断")
    # ★ ref-lang 绝不能随手给个 zh：参考音频是日文、prompt_lang 却报 zh 时，
    #   模型拿到的语义 prompt 是错的，实测输出会塌成 0.68s 的静音（peak 0.003）。
    #   所以默认 auto，按 reference 文本（没有就按目标文本）的字符构成判。
    ap.add_argument("--ref-lang", default="auto", help="参考音频语言；auto=按参考台词自动判")
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--is-half", action="store_true", default=True)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    # ★ speed 是纯时长缩放，不改音高：TTS.py 把它传给 vits_model.decode()，
    #   而 models.py:259 只对 latent 做 F.interpolate 改序列长度（波形采样率不变）。
    #   实测 F0 中位偏移 -1~-3%（噪声量级）⇒ 可以放心拿来调语速。
    #   ★ 不要跟 TTS.py 里那个 `upsample_rate / speed` 混淆 —— 那个只用于算切片边界。
    ap.add_argument(
        "--speed", type=float, default=0.0,
        help="语速系数，1.0=与训练语料同速，越小越慢（0=按 --lang 自动取）。"
             "中文默认 0.85（训练语料是日文，直接念中文约 4.9 字/秒，偏快）。",
    )
    ap.add_argument(
        "--split", default="cut5",
        help="分句策略：cut5=按标点切小句再 5 句一批（默认）· cut0=整段不切。"
             "★ 单句或整段朗读时 cut0 韵律更连贯；但文本过长（>15s）用 cut0 容易中途崩，此时回 cut5。",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repetition-penalty", type=float, default=1.35)
    ap.add_argument(
        "--gap", type=float, default=0.22,
        help="分段合成时，句间插入的静音秒数（默认 0.22）。"
             "★ 上游默认 0.3 且不裁段尾留白，实测叠出 1.2 秒停顿，"
             "长文本听感变成逐句外蹦，这是「日语腔」的一大来源。",
    )
    ap.add_argument(
        "--no-trim", action="store_true",
        help="关闭分段首尾静音裁剪（排障用；正常合成不要关）",
    )
    args = ap.parse_args(argv)

    for p in (args.sovits, args.gpt, args.ref):
        if not os.path.isfile(p):
            print(json.dumps({"ok": False, "error": f"文件不存在: {p}"}, ensure_ascii=False))
            return 2

    # reference 语言：auto 时先看参考台词，再退到目标台词的语言
    ref_lang = args.ref_lang
    ref_lang_auto = False
    if ref_lang in ("auto", "", None):
        ref_lang = guess_lang(args.ref_text, fallback=args.lang)
        ref_lang_auto = True

    t0 = time.time()
    try:
        tts, _cfg = load_torch_pipeline(
            args.gsv_dir, args.version, args.sovits, args.gpt,
            args.device, args.is_half, os.path.join(args.gsv_dir, "_tts_tmp"),
        )
    except Exception as e:                                    # noqa: BLE001
        import traceback
        print(json.dumps({
            "ok": False, "stage": "load",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1500:],
        }, ensure_ascii=False))
        return 3

    import numpy as np
    import torch

    load_sec = time.time() - t0
    vram_after_load = torch.cuda.max_memory_allocated() / 1024 ** 3 if args.device == "cuda" else 0.0

    # 语速：0=按目标语言自动。★ 与 src/config.js 的 LANG_SPEED 保持一致。
    LANG_SPEED = {"zh": 0.85, "ja": 1.0, "en": 0.9, "ko": 1.0, "yue": 0.85}
    speed = args.speed if args.speed and args.speed > 0 else LANG_SPEED.get(args.lang, 1.0)

    inputs = {
        "text": args.text,
        "text_lang": args.lang,
        "ref_audio_path": args.ref,
        "prompt_text": args.ref_text,
        "prompt_lang": ref_lang,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "text_split_method": args.split,
        "batch_size": 1,
        "speed_factor": speed,
        "seed": args.seed,
        "parallel_infer": True,
        "repetition_penalty": args.repetition_penalty,
    }
    # 拿分段而不是整段：只有分段才能逐段裁掉首尾留白，再按合理间隔拼回去
    if not args.no_trim:
        inputs["return_fragment"] = True
        inputs["fragment_interval"] = 0.0

    t1 = time.time()
    sr, audio = None, None
    frags = []
    try:
        for sr, chunk in tts.run(inputs):
            if chunk is None:
                continue
            if args.no_trim:
                audio = chunk                  # 上游已拼好，直接用
            else:
                frags.append(np.asarray(chunk))
    except Exception as e:                                    # noqa: BLE001
        import traceback
        print(json.dumps({
            "ok": False, "stage": "synth",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1500:],
        }, ensure_ascii=False))
        return 4

    if not args.no_trim:
        audio = join_fragments(frags, int(sr), args.gap)
    synth_sec = time.time() - t1

    if audio is None or sr is None:
        print(json.dumps({"ok": False, "stage": "synth", "error": "没有拿到音频"}, ensure_ascii=False))
        return 5

    audio = np.asarray(audio)
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(audio.tobytes())

    dur = len(audio) / float(sr)
    peak = float(np.abs(audio).max()) / 32768.0
    rms = float(np.sqrt(np.mean((audio / 32768.0) ** 2)))

    # 语速自证：按正文字数（去标点）折算字/秒，方便判断「是不是又念快了」。
    # 中文参考：散文朗读 3.8~4.2 字/秒，新闻播报 ~5.0。
    n_char = len([c for c in args.text if not re.match(r"[，。、？！；：…—,\.\?!;:\-\s]", c)])
    cps = round(n_char / dur, 2) if dur > 0 and n_char else 0.0

    # 量化自证：★ peak 和 rms 都**看不出方波**。曾经把 int16 当成 float 做
    #   np.clip(x, -1, 1) 再乘 32767，结果整段音频只剩 {-32767, 0, 32767}
    #   三个取值、89% 样本满幅 —— 听感「失真 + 尖音刺耳」，而 peak 依然是 1.0、
    #   rms 依然正常，所有既有检查全部放行。所以这里必须直接数取值个数。
    uniq = int(np.unique(audio).size)
    clip_ratio = float((np.abs(audio.astype(np.int32)) >= 32700).mean())

    # 产物自证：真的有人声，不是一段静音
    if dur < 0.3 or rms < 1e-4:
        print(json.dumps({
            "ok": False, "stage": "verify",
            "error": f"合成结果异常：duration={dur:.2f}s rms={rms:.6f}",
            "out": os.path.abspath(args.out),
        }, ensure_ascii=False))
        return 6

    # 产物自证（二）：整段被削成方波时，上面的 duration/rms 全都正常，只有取值分布露馅
    if uniq < 64 or clip_ratio > 0.05:
        print(json.dumps({
            "ok": False, "stage": "verify",
            "error": f"音频已失真：唯一取值仅 {uniq} 个，{clip_ratio*100:.1f}% 样本顶到满幅。"
                     f"典型原因是把 int16 当 float 做了 clip(±1) 再乘 32767",
            "out": os.path.abspath(args.out),
        }, ensure_ascii=False))
        return 6

    print(json.dumps({
        "ok": True,
        "out": os.path.abspath(args.out),
        "sampleRate": int(sr),
        "durationSec": round(dur, 2),
        "speed": speed,
        "charsPerSec": cps,
        "uniqValues": uniq,
        "clipRatio": round(clip_ratio, 4),
        "peak": round(peak, 4),
        "rms": round(rms, 5),
        "loadSec": round(load_sec, 1),
        "synthSec": round(synth_sec, 1),
        "vramGB": round(vram_after_load, 2),
        "version": args.version,
        "text": args.text[:60],
        "refLang": ref_lang,
        "refLangAuto": ref_lang_auto,
        "refTextUsed": bool(args.ref_text),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
