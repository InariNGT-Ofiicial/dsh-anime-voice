# -*- coding: utf-8 -*-
"""asr_full.py —— v1：整条一次跑完的 ASR。已知缺陷：长文件提示词衰减。

⚠ 生产用 asr_full2.py（分块 + 每块重新注入提示词）。本脚本只保留两个用途：
  1. 产出 `_full/asr/<ep>.json`，给 asr_full2 找块界（词间最大停顿）用；
  2. 同一时段与 asr_full2 的结果做字符重合比对，挡音乐段幻觉（<0.45 丢）。

缺陷实测：`initial_prompt` 只在开头生效，之后由 condition_on_previous_text 接着上一段
输出走，标点丢了就再也回不来 —— 整集只剩 16 个「。」，且全在头两段。
所以它**不能**当主 ASR：没有标点就只能靠停顿断句，必然切出「つまりここは取り返」这种半句。

输出 `_full/asr/<ep>.json`：words 带相对时间（相对该集正片文件起点）。
"""
import argparse
import json
import os
import sys

DEFAULT_BASE = os.environ.get("ANIME_WORK", os.getcwd())

PROMPT = ("これは、投資の話です。円が上がるのか、それとも下がるのか。"
          "私はそう思う。たぶん、そうなるはずだ。")


def main():
    ap = argparse.ArgumentParser(description="整条一次 ASR（v1，仅供对照；生产用 asr_full2.py）")
    ap.add_argument("--base", default=DEFAULT_BASE, help="工作区根目录（含 _full/raw）")
    ap.add_argument("--eps", default="ep01,ep02,ep03", help="逗号分隔")
    ap.add_argument("--force", action="store_true", help="已有结果也重跑")
    ap.add_argument("--lang", choices=["ja", "zh", "en"], default="ja")
    a = ap.parse_args()

    sys.path.insert(0, os.environ.get("ANIME_EXTRACTOR_SCRIPTS", str(__import__("pathlib").Path(__file__).resolve().parents[2] / "anime-voice-line-extractor/scripts")))
    import anime_lines as AL
    AL._ensure_cuda_dlls()
    from faster_whisper import WhisperModel

    print("加载 whisper ...", flush=True)
    model = WhisperModel(AL.find_whisper_model(None), device="cuda", compute_type="int8_float16")

    for ep in [e.strip() for e in a.eps.split(",") if e.strip()]:
        src = os.path.join(a.base, "_full/raw", ep + ".wav")
        dst = os.path.join(a.base, "_full/asr", ep + ".json")
        if not os.path.isfile(src):
            print(f"{ep} 缺正片 {src}，跳过", flush=True)
            raise SystemExit(f"Missing source: {src}")
        if os.path.isfile(dst) and not a.force:
            print(f"{ep} 已存在，跳过", flush=True)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"\n=== {ep} 转写中（v1 整条一次）===", flush=True)
        segs, info = model.transcribe(
            src, language=a.lang, beam_size=5, vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=True,
            initial_prompt=PROMPT if a.lang == "ja" else None,
            no_speech_threshold=0.6, log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        words = []
        seg_list = []
        dropped = 0
        for s in segs:
            nsp = float(getattr(s, "no_speech_prob", 0.0) or 0.0)
            if nsp > 0.6 and float(getattr(s, "avg_logprob", 0.0) or 0.0) < -1.0:
                dropped += 1
                continue
            if not s.words:
                continue
            seg_list.append(dict(start=s.start, end=s.end, text=s.text.strip(),
                                 nsp=nsp, alp=float(getattr(s, "avg_logprob", 0.0) or 0.0)))
            for w in s.words:
                t = w.word
                if not t.strip():
                    continue
                words.append(dict(start=float(w.start), end=float(w.end),
                                  text=t, conf=float(getattr(w, "probability", 0.0) or 0.0)))
            if len(words) % 3000 < 5:
                print(f"  ... {len(words)} 词 @ {s.end:.0f}s", flush=True)
        json.dump(dict(ep=ep, lang=info.language, words=words, segs=seg_list,
                       dropped=dropped),
                  open(dst, "w", encoding="utf-8"), ensure_ascii=False)
        n_end = sum(1 for w in words if w["text"].strip().endswith(("。", "！", "？")))
        print(f"  完成：{len(words)} 词 / {len(seg_list)} 段 / 丢弃 {dropped} 段 / "
              f"句末标点 {n_end} 个（少是正常的，这是 v1 的缺陷）", flush=True)


if __name__ == "__main__":
    main()
