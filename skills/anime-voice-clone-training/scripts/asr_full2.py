# -*- coding: utf-8 -*-
"""asr_full2.py —— 主 ASR：整集正片分块转写 + 每块重新注入带标点提示词。

v1（asr_full.py）的教训：`initial_prompt` 只在开头有效，跑几十秒后标点就消失
（整集只剩 16 个「。」，且全在头两段）。没有标点又只能靠停顿断句，
等于把旧产线的病带过来了 —— 97.5% 样本无句末标点、66% 以助词或词干收尾。

v2 做法：
  · 把正片按「词间最大停顿」切成约 120 秒的块 —— 块界落在停顿上，不切断句子
  · 每块单独调用，**每块都带 initial_prompt**，标点不会衰减
  · 合并时对块缝做一次修正：块界处间隔 < 0.35s 就把上一块末尾的句号摘掉
    （真句子结束通常有停顿；没有停顿说明是被块切开的）

依赖 `_full/asr/<ep>.json`（v1 的整条结果）：既用来找块界，也是交叉校验的对照版。
输出 `_full/asr/<ep>_p.json`。注意 clip_timestamps 返回的已是整条文件的绝对时间，
不要再加块起点（加了会翻倍）。
"""
import argparse
import json
import os
import sys

DEFAULT_BASE = os.environ.get("ANIME_WORK", os.getcwd())

PROMPT = ("これは、投資の話です。円が上がるのか、それとも下がるのか。"
          "私はそう思う。たぶん、そうなるはずだ。")
CHUNK = 120.0
SEAM_GAP = 0.35


def plan_chunks(prev_words, total, chunk):
    """按词间最大停顿找块界，尽量落在停顿上。"""
    cuts = [0.0]
    t = chunk
    while t < total - 20:
        lo, hi = t - 25, t + 25
        cand = [(prev_words[i + 1]["start"] - prev_words[i]["end"],
                 (prev_words[i]["end"] + prev_words[i + 1]["start"]) / 2)
                for i in range(len(prev_words) - 1)
                if lo <= prev_words[i]["end"] <= hi]
        if cand:
            cand.sort(reverse=True)
            cuts.append(cand[0][1])
        else:
            cuts.append(t)
        t = cuts[-1] + chunk
    cuts.append(total)
    return cuts


def main():
    ap = argparse.ArgumentParser(
        description="整集正片分块 ASR（每块重注提示词）—— 唯一主 ASR")
    ap.add_argument("--base", default=DEFAULT_BASE, help="工作区根目录（含 _full/raw）")
    ap.add_argument("--eps", default="ep01,ep02,ep03", help="逗号分隔")
    ap.add_argument("--chunk", type=float, default=CHUNK, help="块长约多少秒（默认 120）")
    ap.add_argument("--force", action="store_true", help="已有结果也重跑")
    ap.add_argument("--lang", choices=["ja", "zh", "en"], default="ja")
    a = ap.parse_args()

    sys.path.insert(0, os.environ.get("ANIME_EXTRACTOR_SCRIPTS", str(__import__("pathlib").Path(__file__).resolve().parents[2] / "anime-voice-line-extractor/scripts")))
    import soundfile as sf
    import anime_lines as AL
    AL._ensure_cuda_dlls()
    from faster_whisper import WhisperModel

    model = WhisperModel(AL.find_whisper_model(None), device="cuda", compute_type="int8_float16")

    for ep in [e.strip() for e in a.eps.split(",") if e.strip()]:
        src = os.path.join(a.base, "_full", "raw", ep + ".wav")
        base = os.path.join(a.base, "_full", "asr", ep + ".json")
        dst = os.path.join(a.base, "_full", "asr", ep + "_p.json")
        if os.path.isfile(dst) and not a.force:
            print(f"{ep} 已存在，跳过", flush=True)
            continue
        if not os.path.isfile(base):
            raise SystemExit(f"{ep} 缺 v1 结果（先跑 asr_full.py）: {base}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        prev = json.load(open(base, encoding="utf-8"))["words"]
        total = sf.info(src).duration
        cuts = plan_chunks(prev, total, a.chunk)
        print(f"\n=== {ep}  {total:.0f}s 切 {len(cuts)-1} 块 ===", flush=True)

        words, segs, inf = [], [], None
        for k in range(len(cuts) - 1):
            a0, b0 = cuts[k], cuts[k + 1]
            segs_it, inf = model.transcribe(
                src, language=a.lang, beam_size=5, vad_filter=False,
                word_timestamps=True,
                condition_on_previous_text=True,
                initial_prompt=PROMPT if a.lang == "ja" else None,
                no_speech_threshold=0.6, log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                clip_timestamps=[a0, b0],
            )
            n = 0
            for s in list(segs_it):
                if float(getattr(s, "no_speech_prob", 0.0) or 0) > 0.6 and \
                        float(getattr(s, "avg_logprob", 0.0) or 0.0) < -1.0:
                    continue
                # clip 模式返回的是整条文件的绝对时间，不要再加块起点
                segs.append(dict(start=s.start, end=s.end, text=s.text.strip()))
                for w in (s.words or []):
                    if not w.word.strip():
                        continue
                    words.append(dict(start=float(w.start), end=float(w.end),
                                      text=w.word,
                                      conf=float(getattr(w, "probability", 0.0) or 0.0)))
                    n += 1
            print(f"  块 {k+1}/{len(cuts)-1}  {a0:.0f}-{b0:.0f}s  {n} 词", flush=True)

        words.sort(key=lambda x: x["start"])
        # 块缝修正：只在**块界处**摘句号。句中本来就没有停顿的句号是合法的，不能动。
        seams = cuts[1:-1]
        fixed = 0
        for i in range(len(words) - 1):
            if not any(abs(words[i]["end"] - s) < 0.20 for s in seams):
                continue
            if words[i + 1]["start"] - words[i]["end"] < SEAM_GAP:
                t = words[i]["text"].rstrip()
                if t.endswith(("。", "！", "？")):
                    words[i]["text"] = t[:-1]
                    fixed += 1
        n_end = sum(1 for w in words if w["text"].strip().endswith(("。", "！", "？")))
        json.dump(dict(ep=ep, lang=(inf.language if inf else "ja"), words=words,
                       segs=segs, seam_fixed=fixed),
                  open(dst, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  完成：{len(words)} 词 / 句末标点 {n_end} 个 / 块缝摘句号 {fixed} 处",
              flush=True)


if __name__ == "__main__":
    main()
