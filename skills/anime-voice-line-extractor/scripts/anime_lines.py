#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
anime_lines.py — 番剧角色台词拆分管线

从一段番剧音轨里，切出「谁在什么时候说了什么」，
按角色分目录输出音频切片 + 台词文本，并生成 GPT-SoVITS 训练清单。

流程:
    fetch   -> 取音轨 (yt-dlp / 本地文件)
    extract -> ffmpeg 转 16k mono
    vad     -> 语音区间检测 + 合并
    sv      -> cam++ 声纹 embedding
    cluster -> 层次聚类得到说话人簇
    asr     -> faster-whisper 全片转写 (词级时间戳)
    align   -> 词 -> 说话人段 -> 句子
    export  -> 分目录 wav/txt + csv + list + 报告

用法:
    python anime_lines.py run INPUT -o OUT_DIR
    python anime_lines.py rename RUN_DIR --map map.json
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- 控制台
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- 默认路径
DEFAULT_FFMPEG = os.environ.get("FFMPEG_BIN", "")
DEFAULT_HF_HOME = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
DEFAULT_MS_CACHE = os.environ.get("MODELSCOPE_CACHE", str(Path.home() / ".cache/modelscope"))
DEFAULT_WHISPER_ROOT = os.environ.get("ANIME_WHISPER_MODEL", str(Path(DEFAULT_HF_HOME) / "hub/models--Systran--faster-whisper-large-v3/snapshots"))
DEFAULT_CAMPP_ROOT = os.environ.get("ANIME_CAMPP_MODEL", str(Path(DEFAULT_MS_CACHE) / "models/iic--speech_campplus_sv_zh-cn_16k-common/snapshots"))
DEFAULT_YTDLP_PY = os.environ.get("ANIME_YTDLP_PYTHON", sys.executable)
CAMPP_MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"


def find_ffmpeg(explicit: Optional[str] = None) -> str:
    cands = []
    if explicit:
        cands.append(explicit)
    cands += [os.environ.get("FFMPEG_BIN", ""), DEFAULT_FFMPEG, shutil.which("ffmpeg") or ""]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "找不到 ffmpeg。请用 --ffmpeg 指定，或设置环境变量 FFMPEG_BIN。"
    )


def _ensure_cuda_dlls() -> None:
    """
    把 torch/lib 里的 CUDA 运行库显式注册进 DLL 搜索目录。

    CTranslate2（faster-whisper 的推理后端）自己不带 cublas/cudnn，只认进程里
    已可加载的路径。torch 的 lib 目录里就有，但必须先 import torch 并显式加目录，
    否则在「先转写、后声纹」的调用顺序下会报
    `Library cublas64_12.dll is not found or cannot be loaded`。
    """
    if os.name != "nt":
        return
    try:
        import torch  # noqa: F401
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if not os.path.isdir(lib):
            return
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib)
            except OSError:
                pass
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def find_whisper_model(explicit: Optional[str] = None) -> str:
    explicit = explicit or os.environ.get("ANIME_WHISPER_MODEL")
    if explicit and os.path.isdir(explicit):
        # 允许直接给 hub 根目录，自动下潜到 snapshots
        if os.path.isfile(os.path.join(explicit, "model.bin")):
            return explicit
        for sub in ("snapshots",):
            p = os.path.join(explicit, sub)
            if os.path.isdir(p):
                for d in sorted(os.listdir(p)):
                    full = os.path.join(p, d)
                    if os.path.isfile(os.path.join(full, "model.bin")):
                        return full
        raise FileNotFoundError(f"{explicit} 下没找到 model.bin")
    if os.path.isdir(DEFAULT_WHISPER_ROOT):
        for d in sorted(os.listdir(DEFAULT_WHISPER_ROOT)):
            full = os.path.join(DEFAULT_WHISPER_ROOT, d)
            if os.path.isfile(os.path.join(full, "model.bin")):
                return full
    raise FileNotFoundError(
        "找不到 faster-whisper 权重。用 --whisper-model 指定含 model.bin 的目录。"
    )


def find_campp_model(explicit: Optional[str] = None) -> str:
    explicit = explicit or os.environ.get("ANIME_CAMPP_MODEL")
    if explicit and os.path.isdir(explicit):
        if os.path.isfile(os.path.join(explicit, "campplus_cn_common.bin")):
            return explicit
    if os.path.isdir(DEFAULT_CAMPP_ROOT):
        for d in sorted(os.listdir(DEFAULT_CAMPP_ROOT)):
            full = os.path.join(DEFAULT_CAMPP_ROOT, d)
            if os.path.isfile(os.path.join(full, "campplus_cn_common.bin")):
                return full
    return CAMPP_MODEL_ID  # 交给 modelscope 自动下载


# ---------------------------------------------------------------- 数据模型
@dataclass
class SpeechRegion:
    start: float
    end: float
    speaker: int = -1
    confidence: float = 1.0
    reliable: bool = True

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Utterance:
    index: int
    start: float
    end: float
    speaker: int
    text: str
    asr_conf: float
    sv_conf: float
    audio_file: str = ""

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class RunConfig:
    input: str
    out_dir: str
    ffmpeg: str = ""
    whisper_model: str = ""
    campp_model: str = ""
    language: Optional[str] = None
    num_speakers: Optional[int] = None
    cluster_threshold: float = 0.55
    min_sv_dur: float = 1.0
    min_cluster_sec: float = 3.0
    min_line_dur: float = 0.45
    max_line_dur: float = 20.0
    merge_gap: float = 0.35
    split_gap: float = 0.55
    vad_pad: int = 150
    vad_threshold: float = 0.35
    asr_device: str = "cuda"
    asr_compute: str = "int8_float16"
    export_sr: int = 32000
    export_format: str = "wav"
    keep_work: bool = True
    device: str = "cuda"
    keep: str = ""          # 只处理这些区间（跳过 OP/ED/片尾废料），见 parse_keep


# ---------------------------------------------------------------- ffmpeg
def _parse_ts(t: str) -> Optional[float]:
    """支持 12.5 / 1:05 / 1:02:03 三种写法。空串返回 None。"""
    t = t.strip()
    if not t:
        return None
    parts = t.split(":")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"无法解析时间：{t!r}")
    sec = 0.0
    for v in vals:
        sec = sec * 60.0 + v
    return sec


def parse_keep(spec: Optional[str],
               total: Optional[float] = None) -> Optional[List[Tuple[float, float]]]:
    """
    解析「只处理这些区间」的说明，用来跳过 OP / ED / 片尾废料与无对白长尾。

    源片的真实结构往往是「冷开场 + OP + 正片 + ED + 次回预告 + 十几分钟废料」，
    把 OP/ED/废料一起丢给 ASR 既慢又会喂出歌词幻觉。这里只声明**要保留**的区间：

        0-99,171-1324          # 秒
        0:00-1:39,2:51-22:04   # 分:秒
        -99,171-               # 端点留空 = 从片头 / 到片尾

    返回 [(start, end), ...]，按起点排序。
    """
    if not spec or not spec.strip():
        return None
    out: List[Tuple[float, float]] = []
    for chunk in spec.replace("，", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"--keep 段缺少 '-'：{chunk!r}（应形如 0:00-1:39）")
        # 兼容 1:39 里的 ':'，只在最后一个 '-' 处切
        a_s, b_s = chunk.rsplit("-", 1) if chunk.count("-") == 1 else \
            (chunk[:chunk.index("-")], chunk[chunk.index("-") + 1:])
        a = _parse_ts(a_s) or 0.0
        b = _parse_ts(b_s)
        if b is None:
            b = total if total is not None else float("inf")
        if b <= a:
            raise ValueError(f"--keep 段区间非法：{chunk!r}")
        out.append((float(a), float(b)))
    out.sort(key=lambda x: x[0])
    # 合并重叠段
    merged: List[Tuple[float, float]] = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def clip_regions(regions: List[SpeechRegion],
                 keep: List[Tuple[float, float]]) -> List[SpeechRegion]:
    """把语音区间裁剪到 keep 范围内，区间外的整段丢弃。"""
    out: List[SpeechRegion] = []
    for r in regions:
        for a, b in keep:
            s, e = max(r.start, a), min(r.end, b)
            if e - s >= 0.15:
                out.append(SpeechRegion(start=s, end=e, speaker=r.speaker,
                                        confidence=r.confidence,
                                        reliable=r.reliable))
    out.sort(key=lambda x: x.start)
    return out


def clip_stamp_list(keep: List[Tuple[float, float]]) -> List[float]:
    """转成 faster-whisper 的 clip_timestamps（扁平 [a1,b1,a2,b2,...]）。"""
    flat: List[float] = []
    for a, b in keep:
        flat.extend([float(a), float(b)])
    return flat


def run_cmd(cmd: Sequence[str], desc: str = "") -> subprocess.CompletedProcess:
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", creationflags=flags)
    if p.returncode != 0:
        raise RuntimeError(f"{desc} 失败 (code={p.returncode})\n{cmd}\n{p.stderr[-1500:]}")
    return p


def to_wav(src: str, dst: str, ffmpeg: str, sr: int = 16000, mono: bool = True) -> str:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
           "-vn", "-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-c:a", "pcm_s16le", dst]
    run_cmd(cmd, "ffmpeg 转码")
    return dst


def media_duration(path: str, ffmpeg: str) -> float:
    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    exe = ffprobe if os.path.isfile(ffprobe) else ffmpeg
    if os.path.basename(exe).lower().startswith("ffprobe"):
        p = run_cmd([exe, "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nk=1", path], "ffprobe")
        try:
            return float(p.stdout.strip())
        except Exception:
            return 0.0
    p = run_cmd([exe, "-hide_banner", "-i", path], "ffmpeg 探测")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", p.stderr)
    if m:
        h, mi, s = m.groups()
        return int(h) * 3600 + int(mi) * 60 + float(s)
    return 0.0


# ---------------------------------------------------------------- 音频 IO
def read_wav_mono(path: str, target_sr: int = 16000):
    """读 16bit PCM wav -> float32 mono numpy。返回 (sr, array)。"""
    import numpy as np
    with wave.open(path, "rb") as w:
        nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"只支持 16bit PCM，得到 sampwidth={sw}")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    if sr != target_sr:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return sr, a


def write_wav_mono(path: str, audio, sr: int) -> None:
    import numpy as np
    a = np.clip(audio, -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def slice_audio(audio, sr: int, start: float, end: float,
                pad: float = 0.06, fade: float = 0.008):
    """按秒切片，带小 pad 与淡入淡出。"""
    import numpy as np
    s = max(0, int((start - pad) * sr))
    e = min(len(audio), int((end + pad) * sr))
    seg = np.array(audio[s:e], dtype=np.float32, copy=True)
    if seg.size == 0:
        return seg
    nf = max(1, int(fade * sr))
    if seg.size > 2 * nf:
        ramp = np.linspace(0.0, 1.0, nf, dtype=np.float32)
        seg[:nf] *= ramp
        seg[-nf:] *= ramp[::-1]
    return seg


# ---------------------------------------------------------------- VAD
def detect_speech(audio, sr: int = 16000, pad_ms: int = 150,
                  min_speech_ms: int = 200, min_silence_ms: int = 250,
                  threshold: float = 0.35) -> List[SpeechRegion]:
    """
    番剧里 BGM/音效会顶住 VAD 概率，threshold 默认比通用场景低，
    宁可多检一点，后面靠 ASR 置信度和时长再筛。
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    opts = VadOptions(
        threshold=threshold,
        neg_threshold=max(0.0, threshold - 0.15),
        min_speech_duration_ms=min_speech_ms,
        max_speech_duration_s=28,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=pad_ms,
    )
    ts = get_speech_timestamps(audio, opts, sampling_rate=sr)
    return [SpeechRegion(start=t["start"] / sr, end=t["end"] / sr) for t in ts]


def merge_regions(regions: List[SpeechRegion], merge_gap: float,
                  max_dur: float = 12.0) -> List[SpeechRegion]:
    """合并挨得太近的碎段，避免一句台词被切碎。"""
    if not regions:
        return []
    out = [SpeechRegion(regions[0].start, regions[0].end)]
    for r in regions[1:]:
        last = out[-1]
        if r.start - last.end <= merge_gap and (r.end - last.start) <= max_dur:
            last.end = max(last.end, r.end)
        else:
            out.append(SpeechRegion(r.start, r.end))
    return out


# ---------------------------------------------------------------- 声纹
class SpeakerEmbedder:
    """
    cam++ 声纹。批量推理：一次性把所有段算成 fbank，分桶 pad 后一次前向，
    比逐段走 funasr pipeline（每条都要落盘 + 重读）快一个数量级。
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        os.environ.setdefault("MODELSCOPE_CACHE", DEFAULT_MS_CACHE)
        from funasr import AutoModel
        self.pipe = AutoModel(model=model_path, device=device,
                              disable_update=True, disable_pbar=True, disable_log=True)
        self.device = device
        # AutoModel 里的底层网络（CAMPPlus），拿不到就退回逐条推理
        self.net = getattr(self.pipe, "model", None)

    def _fbank(self, audio, sr: int = 16000):
        import torch
        from torchaudio.compliance import kaldi
        t = torch.from_numpy(np.ascontiguousarray(audio)).float()
        f = kaldi.fbank(t.unsqueeze(0), num_mel_bins=80, sample_frequency=sr)
        return f - f.mean(dim=0, keepdim=True)

    def embed_batch(self, segments: List[np.ndarray], sr: int = 16000,
                    batch_size: int = 24) -> np.ndarray:
        """segments: 若干段 float32 mono 音频。返回 (N,192) L2 归一化 embedding。"""
        import torch
        if not segments:
            return np.zeros((0, 192), dtype=np.float32)

        if self.net is None:
            return self._embed_batch_slow(segments, sr)

        feats = [self._fbank(a, sr) for a in segments]
        dim = feats[0].shape[1]
        out = np.zeros((len(feats), 192), dtype=np.float32)

        # 按帧数排序后分桶，桶内长度接近，pad 引入的失真最小
        order = sorted(range(len(feats)), key=lambda i: feats[i].shape[0])
        with torch.no_grad():
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                max_t = max(feats[i].shape[0] for i in idx)
                batch = torch.empty(len(idx), max_t, dim)
                for j, i in enumerate(idx):
                    f = feats[i]
                    n = f.shape[0]
                    batch[j, :n] = f
                    if n < max_t:
                        # pad 用该句帧均值填充，避免 StatsPool 被零点拉偏
                        batch[j, n:] = f.mean(dim=0, keepdim=True)
                e = self.net(batch.to(self.device, dtype=torch.float32))
                e = e.detach().cpu().numpy().reshape(len(idx), -1)
                for j, i in enumerate(idx):
                    out[i] = e[j]
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return out / n

    def _embed_batch_slow(self, segments, sr: int = 16000) -> np.ndarray:
        """兜底：逐条走 funasr pipeline。"""
        out = []
        for a in segments:
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                write_wav_mono(tmp, a, sr)
                r = self.pipe.generate(input=tmp)
                e = r[0]["spk_embedding"]
                if hasattr(e, "cpu"):
                    e = e.cpu().detach().numpy()
                out.append(np.asarray(e).squeeze().astype(np.float32))
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        X = np.vstack(out)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n


# ---------------------------------------------------------------- 聚类
def collapse_small_clusters(labels, cents, durs, min_sec: float):
    """
    把总时长过短的簇并进声纹最接近的大簇。
    番剧里一句话的路人、音效感叹词会各自成簇，留着只会把 roles/ 刷成 20 个目录。
    """
    labels = labels.copy()
    nspk = len(cents)
    tot = np.zeros(nspk, dtype=np.float32)
    for l, d in zip(labels, durs):
        tot[int(l)] += d
    big = [i for i in range(nspk) if tot[i] >= min_sec]
    small = [i for i in range(nspk) if tot[i] < min_sec]
    if not big or not small:
        return labels, cents
    for s in small:
        sims = cents[big] @ cents[s]
        tgt = big[int(np.argmax(sims))]
        labels[labels == s] = tgt
    # 重编号：按规模降序，连续
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts)]
    remap = {int(o): i for i, o in enumerate(order)}
    labels = np.array([remap[int(l)] for l in labels], dtype=int)
    return labels, cents[order]


def cluster_embeddings(embs, durs, threshold: float, num_speakers: Optional[int],
                       min_reliable_dur: float):
    """
    两阶段聚类：先用「时长足够」的可靠样本定簇，再把短样本按最近中心分配。
    返回 (labels, centroids, info)
    """
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    n = len(embs)
    info = {}
    if n == 0:
        return np.array([], dtype=int), np.zeros((0, 0)), info
    if n == 1:
        return np.array([0]), embs[:1], info

    X = np.vstack(embs)
    durs = np.asarray(durs, dtype=np.float32)
    reliable_idx = np.where(durs >= min_reliable_dur)[0]
    if len(reliable_idx) < 4:
        reliable_idx = np.arange(n)
    info["n_reliable"] = int(len(reliable_idx))
    Xr = X[reliable_idx]

    if num_speakers and num_speakers >= 1:
        k = min(num_speakers, len(Xr))
        ac = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        lab_r = ac.fit_predict(Xr)
        info["mode"] = f"fixed_k={k}"
    else:
        ac = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                     metric="cosine", linkage="average")
        lab_r = ac.fit_predict(Xr)
        info["mode"] = f"threshold={threshold}"
        if len(set(lab_r)) < 2 and len(Xr) >= 2:
            # 全部粘在一起时兜底：按最大距离强制二分
            ac2 = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")
            lab_r = ac2.fit_predict(Xr)
            info["mode"] += " (fallback k=2)"

    k = len(set(lab_r))
    # 簇中心
    cents = np.zeros((k, X.shape[1]), dtype=np.float32)
    for c in range(k):
        m = lab_r == c
        if m.any():
            v = Xr[m].mean(axis=0)
            nv = np.linalg.norm(v)
            cents[c] = v / nv if nv > 0 else v

    # 所有样本 -> 最近中心
    sims = X @ cents.T
    labels = sims.argmax(axis=1)
    conf = sims.max(axis=1)

    # 簇号重映射：按规模从大到小排，保证 S01/S02... 连续且主角在前
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts)]
    remap = {int(old): new for new, old in enumerate(order)}
    labels = np.array([remap[int(l)] for l in labels], dtype=int)
    cents = cents[order]
    # 重算置信度（中心已重排）
    sims = X @ cents.T
    conf = sims.max(axis=1)

    info["n_speakers"] = int(len(order))
    info["cluster_sizes"] = [int(c) for c in sorted(counts, reverse=True)]
    info["sv_conf"] = conf
    return labels, cents, info


# ---------------------------------------------------------------- ASR
class Transcriber:
    def __init__(self, model_dir: str, device: str = "cuda", compute_type: str = "int8_float16"):
        from faster_whisper import WhisperModel
        os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
        _ensure_cuda_dlls()
        self.model = WhisperModel(model_dir, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, language: Optional[str] = None,
                   no_speech_threshold: float = 0.6,
                   logprob_threshold: float = -1.0,
                   clip: Optional[List[float]] = None):
        extra = {}
        if clip:
            # 只转写这些区间（跳过 OP/ED/长尾）。输出的时间戳仍是**原片绝对时间**，
            # 这样后续切片能直接对回原视频。
            extra["clip_timestamps"] = clip
        segs, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=False,          # 我们自己做 VAD
            word_timestamps=True,
            condition_on_previous_text=False,   # 关掉，避免幻觉顺着上下文滚雪球
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=logprob_threshold,
            compression_ratio_threshold=2.4,
            **extra,
        )
        words = []
        dropped = 0
        for s in segs:
            # 番剧 OP/ED、纯 BGM 段是幻觉重灾区：高 no_speech 概率 + 低对数概率 -> 整段丢
            nsp = float(getattr(s, "no_speech_prob", 0.0) or 0.0)
            alp = float(getattr(s, "avg_logprob", 0.0) or 0.0)
            if nsp > no_speech_threshold and alp < logprob_threshold:
                dropped += 1
                continue
            if not s.words:
                continue
            for w in s.words:
                t = (w.word or "").strip()
                if not t:
                    continue
                words.append({
                    "start": float(w.start), "end": float(w.end),
                    "text": t, "conf": float(getattr(w, "probability", 0.0) or 0.0),
                })
        self.last_dropped = dropped
        return words, info


# ---------------------------------------------------------------- 对齐
def assign_words_to_speakers(words, regions: List[SpeechRegion],
                             min_overlap: float = 0.05,
                             gap_tolerance: float = 0.25):
    """
    把每个词分配给与它时间重叠最多的语音区间。

    两道门：
    - 与任何区间都无重叠、且相距 > gap_tolerance 的词 → 丢弃（whisper 在 BGM/OP 段的幻觉）
    - 相距在 gap_tolerance 内的词 → 保留。VAD 边界通常比实际发音紧，
      不容这点差会把句首句尾的音节吞掉（「世界的金融危機」被吃成「的金融危機」）。
    """
    out = []
    if not regions:
        return out
    starts = [r.start for r in regions]
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        # 只在 mid 附近 ±6s 的区间里找
        lo = bisect.bisect_left(starts, mid - 6.0)
        hi = bisect.bisect_right(starts, mid + 6.0) + 1
        best, best_ov = None, 0.0
        for k in range(max(0, lo - 1), min(len(regions), hi + 1)):
            r = regions[k]
            ov = min(w["end"], r.end) - max(w["start"], r.start)
            if ov > best_ov:
                best_ov, best = ov, k
        if best is None:
            continue
        r = regions[best]
        near = min(abs(w["end"] - r.start), abs(w["start"] - r.end))
        if best_ov < min_overlap and near > gap_tolerance:
            continue
        out.append({**w, "speaker": r.speaker, "region": best,
                    "sv_conf": r.confidence,
                    "r_start": r.start, "r_end": r.end})
    return out


def _is_degenerate(text: str) -> bool:
    """识别 whisper 的重复幻觉，例如「押せ押せ押せ押せ押せ…」。"""
    t = re.sub(r"\s+", "", text)
    if len(t) < 12:
        return False
    # 同一 2-6 字子串连续重复 >= 3 次
    for n in (2, 3, 4, 5, 6):
        m = re.match(r"^(.{%d})\1{2,}" % n, t)
        if m and len(m.group(0)) >= 12:
            return True
    # 字符熵过低（单一字符占比过高）
    if t:
        top = max(t.count(c) for c in set(t))
        if top / len(t) > 0.55 and len(t) >= 16:
            return True
    return False


def build_utterances(words, split_gap: float, max_dur: float,
                     min_dur: float, trim_to_region: bool = True,
                     drop_degenerate: bool = True) -> List[Utterance]:
    """按 (说话人变化 | 长停顿 | 超长) 切句，并把边界夹回语音区间内。"""
    if not words:
        return []
    groups: List[List[dict]] = []
    cur = [words[0]]
    for w in words[1:]:
        prev = cur[-1]
        same_spk = w["speaker"] == prev["speaker"]
        gap = w["start"] - prev["end"]
        span = w["end"] - cur[0]["start"]
        if (not same_spk) or gap > split_gap or span > max_dur:
            groups.append(cur)
            cur = [w]
        else:
            cur.append(w)
    groups.append(cur)

    utts: List[Utterance] = []
    idx = 0
    for g in groups:
        start, end = g[0]["start"], g[-1]["end"]
        if trim_to_region:
            # 夹回该组所属语音区间，但留 150ms 余量，别把首尾音节切掉
            start = max(start, g[0].get("r_start", start) - 0.15)
            end = min(end, g[-1].get("r_end", end) + 0.15)
        if end - start < min_dur:
            continue
        text = "".join(w["text"] for w in g).strip()
        if not text:
            continue
        if drop_degenerate and _is_degenerate(text):
            continue
        conf = sum(w["conf"] for w in g) / len(g)
        if conf < 0.35:          # 词级置信度过低，基本是幻听
            continue
        sv = sum(w["sv_conf"] for w in g) / len(g)
        idx += 1
        utts.append(Utterance(index=idx, start=start, end=end,
                              speaker=g[0]["speaker"], text=text,
                              asr_conf=conf, sv_conf=sv))
    # 重新编号，保证连续
    for i, u in enumerate(utts, 1):
        u.index = i
    return utts


# ---------------------------------------------------------------- 导出
def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}-{m:02d}-{s:02d}.{ms:03d}"


def safe_name(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", s).strip()
    return s or "unnamed"


def export_all(cfg: RunConfig, utts: List[Utterance], audio, sr: int,
               spk_meta: Dict[int, Dict], out_dir: str, lang: str) -> Dict[str, str]:
    import numpy as np
    roles_dir = os.path.join(out_dir, "roles")
    os.makedirs(roles_dir, exist_ok=True)

    # 清理上轮遗留的角色目录，避免 S11.. 这种旧簇残留在产物里
    old = [d for d in os.listdir(roles_dir)
           if os.path.isdir(os.path.join(roles_dir, d))]
    if old:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        parked = os.path.join(out_dir, "_work", f"roles_prev_{stamp}")
        os.makedirs(parked, exist_ok=True)
        for d in old:
            try:
                shutil.move(os.path.join(roles_dir, d), os.path.join(parked, d))
            except Exception:
                pass

    # 每个 speaker 一个目录
    spk_names: Dict[int, str] = {}
    for sid in sorted({u.speaker for u in utts}):
        name = f"S{sid + 1:02d}"
        spk_names[sid] = name
        os.makedirs(os.path.join(roles_dir, name), exist_ok=True)

    rows = []
    list_lines = []
    for u in utts:
        seg = slice_audio(audio, sr, u.start, u.end)
        if seg.size == 0:
            continue
        base = f"{spk_names[u.speaker]}_{u.index:04d}_{fmt_ts(u.start)}-{fmt_ts(u.end)}"
        wav_path = os.path.join(roles_dir, spk_names[u.speaker], base + ".wav")
        write_wav_mono(wav_path, seg, sr)
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(u.text + "\n")
        u.audio_file = wav_path
        rows.append({
            "index": u.index, "speaker": spk_names[u.speaker],
            "start": round(u.start, 3), "end": round(u.end, 3),
            "dur": round(u.dur, 3), "text": u.text,
            "asr_conf": round(u.asr_conf, 4), "sv_conf": round(u.sv_conf, 4),
            "file": os.path.relpath(wav_path, out_dir).replace("\\", "/"),
        })
        list_lines.append(f"{os.path.abspath(wav_path)}|{spk_names[u.speaker]}|{lang}|{u.text}")

    # lines.csv
    csv_path = os.path.join(out_dir, "lines.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["index", "speaker", "start", "end", "dur", "text",
                            "asr_conf", "sv_conf", "file"])
        w.writeheader()
        w.writerows(rows)

    # lines.jsonl
    jsonl_path = os.path.join(out_dir, "lines.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # speakers.csv
    spk_csv = os.path.join(out_dir, "speakers.csv")
    with open(spk_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "lines", "total_dur", "mean_sv_conf", "mean_asr_conf"])
        for sid in sorted({u.speaker for u in utts}):
            us = [u for u in utts if u.speaker == sid]
            w.writerow([
                spk_names[sid], len(us),
                round(sum(u.dur for u in us), 2),
                round(sum(u.sv_conf for u in us) / len(us), 4),
                round(sum(u.asr_conf for u in us) / len(us), 4),
            ])

    # GPT-SoVITS list
    list_path = os.path.join(out_dir, "gpt_sovits.list")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(list_lines) + ("\n" if list_lines else ""))

    # 角色映射模板
    map_path = os.path.join(out_dir, "speaker_map.json")
    if not os.path.isfile(map_path):
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump({spk_names[s]: spk_names[s] for s in spk_names}, f,
                      ensure_ascii=False, indent=2)

    return {"csv": csv_path, "jsonl": jsonl_path, "speakers": spk_csv,
            "list": list_path, "map": map_path, "roles_dir": roles_dir}


def write_report(out_dir: str, cfg: RunConfig, utts: List[Utterance],
                 regions: List[SpeechRegion], cinfo: Dict, total_dur: float,
                 lang: str, lang_prob: float) -> str:
    lines = []
    A = lines.append
    A("# 番剧角色台词拆分报告\n")
    A(f"- 输入：`{cfg.input}`")
    A(f"- 总时长：{timedelta(seconds=int(total_dur))}")
    speech_dur = sum(r.dur for r in regions)
    A(f"- 检出语音：{len(regions)} 段 / {speech_dur:.1f}s "
      f"（占比 {speech_dur / max(total_dur, 1) * 100:.1f}%）")
    A(f"- 输出台词：{len(utts)} 条 / {sum(u.dur for u in utts):.1f}s")
    A(f"- 识别语言：{lang}（置信度 {lang_prob:.2f}）")
    A(f"- 聚类模式：{cinfo.get('mode', '-')}")
    A(f"- 角色数：{cinfo.get('n_speakers', 0)}")
    A("")

    A("## 角色概览\n")
    A("| 角色 | 台词数 | 总时长(s) | 平均声纹置信 | 平均ASR置信 |")
    A("|---|---|---|---|---|")
    for sid in sorted({u.speaker for u in utts}):
        us = [u for u in utts if u.speaker == sid]
        A(f"| S{sid + 1:02d} | {len(us)} | {sum(u.dur for u in us):.1f} | "
          f"{sum(u.sv_conf for u in us) / len(us):.3f} | "
          f"{sum(u.asr_conf for u in us) / len(us):.3f} |")
    A("")

    low = [u for u in utts if u.sv_conf < 0.5]
    if low:
        A(f"## 需人工复核（{len(low)} 条，声纹置信 < 0.5）\n")
        for u in low[:20]:
            A(f"- [{fmt_ts(u.start)}] S{u.speaker + 1:02d} "
              f"(conf {u.sv_conf:.2f})：{u.text[:40]}")
        if len(low) > 20:
            A(f"- ...另有 {len(low) - 20} 条")
        A("")

    A("## 下一步：给角色起名\n")
    A("1. 听一下 `roles/S01/` 里的样本，确认这是谁。")
    A("2. 编辑 `speaker_map.json`，把 `\"S01\"` 改成 `\"久留美\"` 之类。")
    A("3. 执行：`python anime_lines.py rename <本目录> --map speaker_map.json`")
    A("")
    A("## 落在哪\n")
    A("- `roles/<角色>/` — 每条台词一个 wav + 同名 txt")
    A("- `lines.csv` — 全量索引（时间轴/文本/置信度）")
    A("- `gpt_sovits.list` — 可直接喂 GPT-SoVITS 微调")
    A("- `_work/` — 中间件（16k 分析轨、声纹、段表）")

    p = os.path.join(out_dir, "report.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------- 主流程
def cmd_run(args) -> int:
    t_all = time.time()
    ffmpeg = find_ffmpeg(args.ffmpeg)
    src = os.path.abspath(args.input)
    if not os.path.isfile(src):
        log(f"[x] 输入不存在：{src}")
        return 2

    out_dir = os.path.abspath(args.out_dir)
    work = os.path.join(out_dir, "_work")
    os.makedirs(work, exist_ok=True)

    cfg = RunConfig(
        input=src, out_dir=out_dir, ffmpeg=ffmpeg,
        whisper_model=args.whisper_model or "", campp_model=args.campp_model or "",
        language=args.language, num_speakers=args.num_speakers,
        cluster_threshold=args.cluster_threshold, min_sv_dur=args.min_sv_dur,
        min_cluster_sec=args.min_cluster_sec,
        min_line_dur=args.min_line_dur, max_line_dur=args.max_line_dur,
        merge_gap=args.merge_gap, split_gap=args.split_gap, vad_pad=args.vad_pad,
        vad_threshold=args.vad_threshold,
        asr_device=args.asr_device, asr_compute=args.asr_compute,
        export_sr=args.export_sr, keep_work=not args.drop_work, device=args.device,
        keep=args.keep or "",
    )

    # 1. 分析轨（16k）与导出轨（export_sr）
    wav16 = os.path.join(work, "audio_16k.wav")
    log("[1/7] 提取音轨 16k ...")
    to_wav(src, wav16, ffmpeg, 16000)
    total_dur = media_duration(src, ffmpeg) or 0.0
    if total_dur <= 0:
        _, _a = read_wav_mono(wav16)
        total_dur = len(_a) / 16000.0
    log(f"      时长 {timedelta(seconds=int(total_dur))}")

    wav_exp = os.path.join(work, f"audio_{cfg.export_sr}.wav")
    if cfg.export_sr != 16000:
        log(f"[2/7] 准备导出轨 {cfg.export_sr}Hz ...")
        to_wav(src, wav_exp, ffmpeg, cfg.export_sr)
    else:
        wav_exp = wav16

    sr, audio = read_wav_mono(wav16)

    # 2. VAD
    log("[3/7] 语音区间检测 ...")
    regions = detect_speech(audio, sr, pad_ms=cfg.vad_pad,
                            threshold=cfg.vad_threshold)
    regions = merge_regions(regions, cfg.merge_gap, max_dur=cfg.max_line_dur)
    keep_ranges = parse_keep(cfg.keep or None, total=total_dur)
    if keep_ranges:
        before = len(regions)
        regions = clip_regions(regions, keep_ranges)
        skipped = total_dur - sum(b - a for a, b in keep_ranges)
        log(f"      按 --keep 跳过 {skipped:.0f}s（OP/ED/长尾）："
            f"{before} -> {len(regions)} 段")
    speech_dur = sum(r.dur for r in regions)
    log(f"      {len(regions)} 个语音段，合计 {speech_dur:.1f}s "
        f"（占全片 {speech_dur / max(total_dur, 1) * 100:.0f}%）")
    if not regions:
        log("[!] 没检出语音，退出。")
        return 3

    # 3. 声纹
    log("[4/7] 声纹 embedding ...")
    emb = SpeakerEmbedder(find_campp_model(cfg.campp_model or None), device=cfg.device)
    segs, durs, keep = [], [], []
    for i, r in enumerate(regions):
        if r.dur < 0.3:
            continue
        seg = slice_audio(audio, sr, r.start, r.end, pad=0.0)
        if seg.size < int(0.2 * sr):
            continue
        segs.append(seg)
        durs.append(r.dur)
        keep.append(i)
    if not segs:
        log("[x] 没有可用的语音段。")
        return 4
    log(f"      批量推理 {len(segs)} 段 ...")
    embs = emb.embed_batch(segs, sr)
    regions = [regions[i] for i in keep]
    log(f"      完成 {len(embs)} 段，维度 {embs.shape[1]}")

    # 4. 聚类
    log("[5/7] 说话人聚类 ...")
    labels, cents, cinfo = cluster_embeddings(
        embs, durs, cfg.cluster_threshold, cfg.num_speakers, cfg.min_sv_dur)
    n_before = cinfo["n_speakers"]
    if cfg.min_cluster_sec > 0:
        labels, cents = collapse_small_clusters(
            labels, cents, np.asarray(durs), cfg.min_cluster_sec)
        cinfo["n_speakers"] = int(len(cents))
        cinfo["collapsed"] = n_before - len(cents)
    confs = cinfo.pop("sv_conf", None)
    if confs is None or cinfo.get("collapsed"):
        sims = np.vstack(embs) @ cents.T
        confs = sims.max(axis=1)
    for i, r in enumerate(regions):
        r.speaker = int(labels[i])
        r.reliable = durs[i] >= cfg.min_sv_dur
        if confs is not None:
            r.confidence = float(confs[i])
    log(f"      得到 {cinfo['n_speakers']} 个说话人（{cinfo['mode']}）")

    # 5. ASR
    log("[6/7] 语音识别 ...")
    tr = Transcriber(find_whisper_model(cfg.whisper_model or None),
                     device=cfg.asr_device, compute_type=cfg.asr_compute)
    words, ainfo = tr.transcribe(
        wav16, language=cfg.language,
        clip=clip_stamp_list(keep_ranges) if keep_ranges else None)
    # 存原始词级时间戳：refine 阶段要按词重判说话人，避免重跑 ASR
    with open(os.path.join(work, "words.json"), "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)
    lang = getattr(ainfo, "language", "unknown") or "unknown"
    lang_prob = float(getattr(ainfo, "language_probability", 0.0) or 0.0)
    log(f"      {len(words)} 个词，语言 {lang} ({lang_prob:.2f})，"
        f"丢弃静音段 {getattr(tr, 'last_dropped', 0)} 个")

    # 6. 对齐 + 切句
    log("[7/7] 对齐与导出 ...")
    words = assign_words_to_speakers(words, regions)
    utts = build_utterances(words, cfg.split_gap, cfg.max_line_dur, cfg.min_line_dur)
    log(f"      {len(utts)} 条台词")

    sr_e, audio_e = read_wav_mono(wav_exp)
    spk_meta: Dict[int, Dict] = {}
    paths = export_all(cfg, utts, audio_e, sr_e, spk_meta, out_dir, lang)

    # 段表 & 声纹存盘
    np.savez(os.path.join(work, "sv_embeddings.npz"),
             X=np.vstack(embs), labels=labels, durs=np.asarray(durs))
    with open(os.path.join(work, "regions.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in regions], f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "detected_language": lang}, f,
                  ensure_ascii=False, indent=2)

    rep = write_report(out_dir, cfg, utts, regions, cinfo, total_dur, lang, lang_prob)

    log("")
    log("=" * 58)
    log(f"完成，用时 {time.time() - t_all:.0f}s")
    log(f"  角色：{cinfo['n_speakers']}  台词：{len(utts)} 条")
    log(f"  目录：{paths['roles_dir']}")
    log(f"  索引：{paths['csv']}")
    log(f"  清单：{paths['list']}")
    log(f"  报告：{rep}")
    log("=" * 58)
    if not cfg.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return 0


# ---------------------------------------------------------------- 词级判定
def load_role_map(path: str) -> Dict[str, List[str]]:
    """读角色映射，支持两种写法：

        {"S01": "久留美", "S04": "久留美"}           # 簇 -> 角色（可多对一）
        {"久留美": ["S01", "S04"], "父亲": ["S02"]}  # 角色 -> 簇列表

    统一返回 {角色名: [簇键, ...]}。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    roles: Dict[str, List[str]] = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)):
            name, keys = str(k), [str(x) for x in v]
        else:
            name, keys = str(v), [str(k)]
        name = safe_name(name)
        roles.setdefault(name, [])
        for kk in keys:
            kk = str(kk).strip()
            if kk and kk not in roles[name]:
                roles[name].append(kk)
    return {n: ks for n, ks in roles.items() if n and n.lower() != "none"}


def _cluster_key_to_idx(k: str) -> Optional[int]:
    m = re.fullmatch(r"[sS](\d+)", str(k).strip())
    return int(m.group(1)) - 1 if m else None


def word_level_diarize(words: List[dict], audio, sr: int, emb,
                       role_cents: Dict[str, Any], min_win: float = 1.0,
                       hop: float = 0.25, floor: float = 0.30,
                       accept: float = 0.52, margin: float = 0.06,
                       min_run: float = 0.75,
                       fill_gap: float = 0.70) -> List[dict]:
    """
    滑窗说话人判定 + 迟滞，再把结果映射回词。

    为什么不是逐词判定：词的时长常在 0.2~0.5s，声纹在这个尺度上噪声极大，
    实测同一人被逐词判定时会平均每 9 个词翻一次标签，句子被切成单字碎片。
    而直接取 argmax 还会把「谁的相似度都不高」的噪声词硬塞给最像的人
    （实测「いやいや大丈夫」只有 0.27~0.39 相似度，却被判给了主角）。

    所以改成：
    1. 以 min_win 为窗、hop 为步长做滑窗，每窗独立判定；
    2. 迟滞——新角色必须同时满足 sim >= accept 且领先当前角色 margin 才切换；
    3. 短于 min_run 秒的异类段视为抖动，并回邻居；
    4. 最高相似度低于 floor 的窗判为 None（非语音/噪声），不硬分配。

    这样边界只落在证据真正翻转的地方，碎片和误并同时被压下去。
    """
    import numpy as np
    n = len(words)
    if n == 0 or not role_cents:
        return []

    total = len(audio) / float(sr)
    names = list(role_cents.keys())
    C = np.vstack([role_cents[k] for k in names])
    accept = max(accept, floor)

    t0 = max(0.0, float(words[0]["start"]) - 0.5)
    t1 = min(total, float(words[-1]["end"]) + 0.5)
    half = min_win / 2.0
    centers, segs = [], []
    t = t0 + half
    while t < t1:
        a, b = max(0.0, t - half), min(total, t + half)
        if b - a >= 0.4:
            seg = slice_audio(audio, sr, a, b, pad=0.0)
            if seg.size >= int(0.3 * sr):
                centers.append(t)
                segs.append(seg)
        t += hop
    if not segs:
        return []

    E = emb.embed_batch(segs, sr)
    S = np.asarray(E) @ C.T

    # 只信高置信窗：证据不足就留空，交给后面的空洞回填，而不是硬沿用上一个
    # 判定。沿用会让「一次判错」顺着时间轴一路扩散（实测把整段双人对话都
    # 并进了主角）。
    wl: List[Optional[int]] = []
    wsim: List[float] = []
    for v in S:
        best = int(v.argmax())
        s_best = float(v[best])
        wl.append(best if s_best >= accept else None)
        wsim.append(s_best)

    # 空洞回填：短于 fill_gap 的连续空白，两侧同角色则填回（气声/咬字不清）
    fill_frames = max(1, int(round(fill_gap / hop)))
    i = 0
    while i < len(wl):
        if wl[i] is not None:
            i += 1
            continue
        j = i
        while j < len(wl) and wl[j] is None:
            j += 1
        if 0 < i and j < len(wl) and wl[i - 1] == wl[j] and (j - i) <= fill_frames:
            for k in range(i, j):
                wl[k] = wl[i - 1]
        i = j

    # 短段吸收：短于 min_run 的连续同标签段并回邻居（抖动）
    min_frames = max(1, int(round(min_run / hop)))
    changed, guard = True, 0
    while changed and guard < 8:
        changed = False
        guard += 1
        i = 0
        while i < len(wl):
            j = i
            while j < len(wl) and wl[j] == wl[i]:
                j += 1
            if (j - i) < min_frames and 0 < i and j < len(wl) \
                    and wl[i - 1] == wl[j] and wl[i - 1] is not None:
                for k in range(i, j):
                    wl[k] = wl[i - 1]
                changed = True
            i = j

    out = [dict(w) for w in words]
    for i, w in enumerate(out):
        mid = (float(w["start"]) + float(w["end"])) / 2.0
        k = int(round((mid - (t0 + half)) / hop))
        k = max(0, min(len(wl) - 1, k))
        lab, sv = wl[k], wsim[k]
        out[i].update({
            "role": names[lab] if lab is not None else None,
            "role_sim": float(sv),
            "role_margin": 0.0,
            # 保留全量相似度：模糊词要靠它判断「更像谁」，不能只凭位置归属
            "sims": {names[j]: float(S[k][j]) for j in range(len(names))},
        })
    winfo = {"centers": centers, "labels": wl, "hop": hop, "win": min_win}
    return out, winfo


def _best_role(ws: List[dict]) -> Optional[str]:
    acc: Dict[str, float] = {}
    for w in ws:
        for n, s in (w.get("sims") or {}).items():
            acc[n] = acc.get(n, 0.0) + s
    return max(acc, key=acc.get) if acc else None


def split_by_diarization(words: List[dict], split_gap: float, max_dur: float,
                         min_dur: float, edge_blur: float = 0.50,
                         drop_degenerate: bool = True) -> List[Utterance]:
    """
    按 diarization 结果切句，只保留「角色明确」的句子。

    规则（严苛版）：
    - 明确属于其他角色的词 -> 硬边界；
    - 证据不足（role 为 None）的连续段：
        * 落在句首/句尾且短于 edge_blur -> 当作本角色咬字不清，一并保留；
        * 更长或夹在句中 -> 视为他人话语，硬边界。
      这样既不会吞掉句首音节（「ニュースは」的「ニュー」常被判模糊），
      也不会把「いや近いし大丈夫」这种完整的一句别人的话留在切片里。
    - 相邻同角色段若间隔 <= split_gap 则合并。
    """
    if not words:
        return []

    runs: List[list] = []          # [label, start, end, [words]]
    for w in words:
        lab = w.get("role")
        if runs and runs[-1][0] == lab and \
                float(w["start"]) - runs[-1][2] <= split_gap:
            runs[-1][2] = float(w["end"])
            runs[-1][3].append(w)
        else:
            runs.append([lab, float(w["start"]), float(w["end"]), [w]])

    # 处理模糊段：短的并入相邻明确段
    fixed: List[list] = []
    for i, r in enumerate(runs):
        if r[0] is not None:
            fixed.append(list(r))
            continue
        dur = r[2] - r[1]
        prev = fixed[-1] if fixed else None
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        if dur <= edge_blur:
            # 短模糊段不能无脑并进前一句：前一句可能是别人，会把首字一起陪葬
            # （「ニュースは」的「ニュ」就这样丢过）。优先按声纹更像谁归属。
            best = _best_role(r[3])
            prev_ok = prev is not None and prev[0] is not None
            next_ok = nxt is not None and nxt[0] is not None
            tgt = None
            if prev_ok and prev[0] == best:
                tgt = prev
            elif next_ok and nxt[0] == best:
                tgt = nxt
            elif prev_ok and next_ok:
                tgt = prev if (r[1] - prev[2]) <= (nxt[1] - r[2]) else nxt
            elif prev_ok:
                tgt = prev
            elif next_ok:
                tgt = nxt
            if tgt is not None:
                if tgt is prev:
                    prev[2] = r[2]
                    prev[3].extend(r[3])
                else:
                    nxt[1] = r[1]
                    nxt[3] = r[3] + nxt[3]
                continue
        # 长的模糊段：保留为 None，后面整体丢弃
        fixed.append(list(r))

    # 合并被模糊段隔开的同类段
    merged: List[list] = []
    for r in fixed:
        if merged and r[0] is not None and merged[-1][0] == r[0] and \
                r[1] - merged[-1][2] <= split_gap:
            merged[-1][2] = r[2]
            merged[-1][3].extend(r[3])
        else:
            merged.append(r)

    utts: List[Utterance] = []
    for lab, st, en, ws in merged:
        if lab is None:
            continue
        # 超长句子按词边界切开：一口气 19 秒的切片，中间必然藏着气口和停顿，
        # 对训练和试听都不友好。
        chunks: List[List[dict]] = []
        cur: List[dict] = []
        head = None
        for x in ws:
            if cur and (float(x["end"]) - head) > max_dur:
                chunks.append(cur)
                cur, head = [], None
            if head is None:
                head = float(x["start"])
            cur.append(x)
        if cur:
            chunks.append(cur)

        for ch in chunks:
            cst, cen = float(ch[0]["start"]), float(ch[-1]["end"])
            if cen - cst < min_dur:
                continue
            text = "".join(x["text"] for x in ch).strip()
            if not text or (drop_degenerate and _is_degenerate(text)):
                continue
            conf = sum(x["conf"] for x in ch) / len(ch)
            if conf < 0.35:
                continue
            sv = sum(x.get("role_sim", 0.0) for x in ch) / len(ch)
            utts.append(Utterance(index=0, start=cst, end=cen, speaker=lab,
                                  text=text, asr_conf=conf, sv_conf=sv))
    for i, u in enumerate(utts, 1):
        u.index = i
    return utts


_BOILERPLATE_PAT = re.compile(
    r"ご視聴ありがとう|ご覧いただき|ご覧くださり|チャンネル登録|高評価|"
    r"次回予告|次回もお楽しみに|提供[：:]|スポンサー|"
    r"本編は|ここからが本編|"
    # 版权警示画面上 whisper 的幻觉：整句被 VAD 切成两半时，
    # 只剩「ご視」/「ありがとうございました」这种孤立片段，单独也要拦。
    r"^(?:ご視|ご視聴|ありがとうございました|ありがとうございます)[。、!?！？]?$"
)


def _is_boilerplate(t: str) -> bool:
    """
    频道套话 / 片头提示语。

    whisper 在版权警示画面、OP 前的静音段上会幻觉出「ご視聴ありがとうございました」
    这类完整通顺的句子——它既不是重复串也不是低语速，能躲过前面所有判据，
    但绝不可能是角色台词。
    """
    return bool(_BOILERPLATE_PAT.search(t or ""))


def fill_edge_words(words: List[dict], max_run: int = 2) -> List[dict]:
    """
    回填句子首尾的未判定词。

    逐词声纹的窗口以词为中心向两侧扩，句首/句末词的窗口一大半落在静音上，
    声纹质量极差 → 判不出来 → 被丢弃 → 句子开头被咬掉一个字：
        「絶対に手をつけちゃいけない」→「対に手をつけちゃいけない」
        「失敗したら完全に終わり」  →「したら完全に終わり」
    规则：连续 ≤max_run 个未判定的词，若前后同角色（或贴着句首/句末），归给该角色。
    真正的他人语音会在 strict_recheck 用滑窗占比再筛一遍。
    """
    n = len(words)
    out = [dict(w) for w in words]
    i = 0
    while i < n:
        if out[i].get("role"):
            i += 1
            continue
        j = i
        while j < n and not out[j].get("role"):
            j += 1
        prev = out[i - 1].get("role") if i > 0 else None
        nxt = out[j].get("role") if j < n else None
        run = j - i
        if run <= max_run and prev and (prev == nxt or nxt is None):
            for k in range(i, j):
                out[k]["role"] = prev
                out[k]["role_filled"] = True
        elif run <= max_run and nxt and (prev is None or prev == nxt):
            for k in range(i, j):
                out[k]["role"] = nxt
                out[k]["role_filled"] = True
        i = j
    return out


def strict_recheck(utts: List[Utterance], winfo: Dict[str, Any],
                   target: str, names: List[str],
                   max_marks: int = 1,
                   tol: float = 0.15,
                   near: Sequence[int] = (),
                   other_ratio: float = 0.34) -> List[Utterance]:
    """
    最严一道闸，三道判据，命中即整条丢弃：

    1. 句内「他人语音」的滑窗**占句长比例**超过 other_ratio（默认 34%）；
       单点误判只占 10% 左右，不会误杀，但真有别人插话必然超过这个比例。
    2. 句内句末标点（? ！）超过 max_marks 个——这是「一问一答被并成一句」的
       硬信号。同学的声音和主角可能极像（实测相似度 0.5~0.63），声纹在这种情况下
       失效，但两个人来回说话必然留下多个问号/感叹号。
    3. 频道套话（「ご視聴ありがとうございました」）——whisper 在版权画面/静音上的
       幻觉，通顺到能躲过所有重复性判据，但绝不可能是角色台词。

    near：与 target 声纹相似度极高的角色（如同一部剧里的同学、母女）。这类角色
    在词级上本就分不开，若把它们当作「他人」，全剧会被砍掉大半，故豁免。
    """
    centers = winfo.get("centers") or []
    labels = winfo.get("labels") or []
    hop = winfo.get("hop", 0.25)
    tgt = names.index(target) if target in names else -1
    bad_ids = set(range(len(names))) - ({tgt} if tgt >= 0 else set()) - set(near)
    keep = []
    drop_other = drop_marks = drop_slow = drop_boiler = 0
    for u in utts:
        if centers and labels and bad_ids:
            hit = sum(1 for c, lab in zip(centers, labels)
                      if lab in bad_ids and u.start - hop <= c <= u.end + hop)
            total = max(1.0, (u.end - u.start) / hop + 2.0)
            if hit / total > other_ratio:
                drop_other += 1
                continue
        if _is_boilerplate(u.text):
            drop_boiler += 1
            continue
        marks = sum(u.text.count(ch) for ch in "?？!！")
        if marks > max_marks:
            drop_marks += 1
            continue
        # 语速异常：10 秒只出 4 个字，基本是标题卡音效/BGM 被 VAD 当成语音
        chars = len(re.sub(r"[\s、。,.!?？！」』…\u3000]", "", u.text))
        if u.dur >= 4.0 and chars / max(u.dur, 0.1) < 1.0:
            drop_slow += 1
            continue
        keep.append(u)
    if drop_other or drop_marks or drop_slow or drop_boiler:
        log(f"      严苛复核：丢弃 {drop_other} 条含他人语音、"
            f"{drop_marks} 条多轮对话、{drop_slow} 条异常语速、"
            f"{drop_boiler} 条频道套话")
    for i, u in enumerate(keep, 1):
        u.index = i
    return keep


def merge_short_utterances(utts: List[Utterance], min_dur: float = 0.7,
                           join_gap: float = 0.5) -> List[Utterance]:
    """把过短的句子并进相邻同角色句，避免单字成条。"""
    if not utts:
        return utts
    utts = sorted(utts, key=lambda u: u.start)
    res: List[Utterance] = []
    for u in utts:
        if res:
            p = res[-1]
            gap = u.start - p.end
            short = u.dur < min_dur or p.dur < min_dur
            if p.speaker == u.speaker and gap <= join_gap and short \
                    and (u.end - p.start) <= 20.0:
                p.end = u.end
                p.text = (p.text + u.text).strip()
                p.asr_conf = (p.asr_conf + u.asr_conf) / 2.0
                p.sv_conf = (p.sv_conf + u.sv_conf) / 2.0
                continue
        res.append(Utterance(index=0, start=u.start, end=u.end,
                             speaker=u.speaker, text=u.text,
                             asr_conf=u.asr_conf, sv_conf=u.sv_conf))
    for i, u in enumerate(res, 1):
        u.index = i
    return res



def strip_silence(seg, sr: int, min_gap: float = 0.35, edge: float = 0.08,
                  ratio: float = 0.055):
    """压掉切片内部的长静默（保留 edge 秒余量），别让训练音频里挂着大片空白。"""
    if seg is None or len(seg) == 0:
        return seg
    frame = max(1, int(0.02 * sr))
    nf = len(seg) // frame
    if nf < 5:
        return seg
    body = seg[:nf * frame].reshape(nf, frame).astype(np.float32)
    rms = np.sqrt((body ** 2).mean(axis=1) + 1e-12)
    peak = float(rms.max())
    if peak <= 1e-7:
        return seg
    voiced = rms > peak * ratio

    runs, i = [], 0
    while i < nf:
        if voiced[i]:
            j = i
            while j < nf and voiced[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return seg

    gap_f = int(min_gap * sr) // frame
    keep_f = int(edge * sr) // frame
    pieces, prev_end = [], None
    for a, b in runs:
        if prev_end is not None and (a - prev_end) < gap_f:
            # 静默不构成断点，补上原始片段保持连续
            pieces.append(seg[prev_end * frame: b * frame])
        else:
            pieces.append(seg[max(0, a - keep_f) * frame: min(nf, b + keep_f) * frame])
        prev_end = b
    if not pieces:
        return seg
    out = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
    return out.astype(seg.dtype)


# ---------------------------------------------------------------- refine
def cmd_refine(args) -> int:
    """
    用角色映射表重做词级归属，只导出目标角色，并清洗切片内的他人话语。

    复用 run 留下的 _work（音频 / 声纹 / 段表 / 词级时间戳），不重跑 ASR。
    """
    t_all = time.time()
    run_dir = os.path.abspath(args.run_dir)
    work = os.path.join(run_dir, "_work")
    missing = [f for f in ("regions.json", "sv_embeddings.npz", "audio_16k.wav")
               if not os.path.isfile(os.path.join(work, f))]
    if missing:
        log(f"[x] _work 缺文件：{', '.join(missing)}")
        log("    refine 依赖 run 的中间产物，请先完整跑一次 run（不要加 --drop-work）")
        return 2

    roles_map = load_role_map(args.map)
    if not roles_map:
        log("[x] 映射表是空的")
        return 2

    z = np.load(os.path.join(work, "sv_embeddings.npz"))
    X, labels = z["X"], z["labels"].astype(int)

    # 未在映射表里出现的簇单独成角色，避免被硬塞进目标角色
    mapped_idx = set()
    for keys in roles_map.values():
        for k in keys:
            i = _cluster_key_to_idx(k)
            if i is not None:
                mapped_idx.add(i)
    for k in sorted(set(labels.tolist())):
        if k not in mapped_idx:
            nm = f"S{k + 1:02d}"
            roles_map[nm] = [nm]

    role_cents: Dict[str, Any] = {}
    for name, keys in roles_map.items():
        idxs = [i for i in (_cluster_key_to_idx(k) for k in keys) if i is not None]
        if not idxs:
            log(f"[warn] {name}：簇键无法解析 {keys}")
            continue
        m = np.isin(labels, idxs)
        if not m.any():
            log(f"[warn] {name}：簇 {keys} 在声纹表里不存在")
            continue
        v = X[m].sum(axis=0)
        role_cents[name] = v / (np.linalg.norm(v) + 1e-9)
    if not role_cents:
        log("[x] 一个有效角色都没有")
        return 3
    log(f"角色中心：{len(role_cents)} 个 -> {', '.join(role_cents)}")

    target = safe_name(args.target) if args.target else None
    if target and target not in role_cents:
        log(f"[x] 目标角色 {target} 不在映射表里，可选：{', '.join(role_cents)}")
        return 3

    sr, audio = read_wav_mono(os.path.join(work, "audio_16k.wav"))
    wpath = os.path.join(work, "words.json")
    if os.path.isfile(wpath):
        with open(wpath, encoding="utf-8") as f:
            raw_words = json.load(f)
    else:
        # 旧产物没有词级时间戳，现场补一次转写（比重跑整条管线便宜得多）
        log("未找到 words.json，补跑一次词级转写 ...")
        rc0 = os.path.join(run_dir, "run_config.json")
        wm, lg, keep_spec = "", None, ""
        if os.path.isfile(rc0):
            try:
                with open(rc0, encoding="utf-8") as f:
                    j0 = json.load(f)
                wm = j0.get("whisper_model") or ""
                lg = j0.get("detected_language") or j0.get("language")
                keep_spec = j0.get("keep") or ""
            except Exception:
                pass
        _kr = parse_keep(keep_spec or None)
        tr = Transcriber(find_whisper_model(wm or None), device=args.device)
        raw_words, _ = tr.transcribe(os.path.join(work, "audio_16k.wav"),
                                     language=lg,
                                     clip=clip_stamp_list(_kr) if _kr else None)
        with open(wpath, "w", encoding="utf-8") as f:
            json.dump(raw_words, f, ensure_ascii=False)
    log(f"词级时间戳 {len(raw_words)} 个，开始逐词判定 ...")

    # 先用 run 的 VAD 段表过一遍：落在语音区间外的词是 whisper 在 OP/BGM 段
    # 的幻觉（歌词、无意义重复），必须在判定前就扔掉。
    with open(os.path.join(work, "regions.json"), encoding="utf-8") as f:
        spans = [(float(r["start"]), float(r["end"]))
                 for r in json.load(f)]
    spans.sort()
    st = [s for s, _ in spans]
    tol = 0.55
    kept_words = []
    for w in raw_words:
        mid = (float(w["start"]) + float(w["end"])) / 2.0
        k = bisect.bisect_right(st, mid + tol) - 1
        hit = False
        for j in (k, k - 1):
            if 0 <= j < len(spans):
                a, b = spans[j]
                if a - tol <= mid <= b + tol:
                    hit = True
                    break
        if hit:
            kept_words.append(w)
    if len(kept_words) < len(raw_words):
        log(f"      语音区间外丢弃 {len(raw_words) - len(kept_words)} 词（幻觉/歌词）")
    raw_words = kept_words

    emb = SpeakerEmbedder(find_campp_model(args.campp_model or None),
                          device=args.device)
    words, winfo = word_level_diarize(raw_words, audio, sr, emb, role_cents,
                                      min_win=args.word_win, hop=args.word_hop,
                                      floor=args.word_floor,
                                      accept=args.word_accept,
                                      margin=args.word_margin,
                                      min_run=args.word_min_run,
                                      fill_gap=args.word_fill_gap)
    words = fill_edge_words(words)
    ok = sum(1 for w in words if w.get("role"))
    log(f"      明确判定 {ok}/{len(words)} 词（含边界回填）")
    if not ok:
        log("[x] 没有可判定的词")
        return 4

    rnames = list(role_cents.keys())
    utts = split_by_diarization(words, args.split_gap, args.max_line_dur,
                                args.min_line_dur, edge_blur=args.edge_blur)
    log(f"      切句：{len(utts)} 条")

    # 统计 & 只留目标
    from collections import Counter
    cnt = Counter(u.speaker for u in utts)
    log("      分布：" + "，".join(f"{k} {v} 条" for k, v in
                                   sorted(cnt.items(), key=lambda kv: -kv[1])))

    if target and args.strict:
        # 与目标角色声纹过于接近的角色（同学、母女等）不当作「他人」，
        # 否则词级判定在两人之间摇摆时会被全剧误杀。
        near: List[int] = []
        if target in rnames and len(rnames) > 1:
            C = np.vstack([role_cents[k] for k in rnames])
            sim = C @ C[rnames.index(target)]
            near = [i for i, s in enumerate(sim)
                    if rnames[i] != target and float(s) >= args.near_sim]
            if near:
                log("      近似角色（相似度 ≥ %.2f，不视为他人）：%s"
                    % (args.near_sim,
                       "，".join(f"{rnames[i]} {float(sim[i]):.2f}" for i in near)))
        utts = strict_recheck(utts, winfo, target, rnames,
                              max_marks=args.max_marks, near=near)

    if target:
        utts = [u for u in utts if u.speaker == target]

    # 用导出轨切音频（音质更好）
    sr_e, audio_e = read_wav_mono(os.path.join(work, f"audio_{args.export_sr}.wav")) \
        if os.path.isfile(os.path.join(work, f"audio_{args.export_sr}.wav")) \
        else (sr, audio)

    lang = args.language or "ja"
    rc = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(rc) and not args.language:
        try:
            with open(rc, encoding="utf-8") as f:
                rcj = json.load(f)
            lang = rcj.get("detected_language") or rcj.get("language") or lang
        except Exception:
            pass

    tag = target or "all"
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else \
        os.path.join(run_dir, f"clean_{safe_name(tag)}")
    roles_dir = os.path.join(out_dir, "roles")
    os.makedirs(roles_dir, exist_ok=True)
    # 只清本角色的旧产物，别动别的角色
    for d in os.listdir(roles_dir):
        p = os.path.join(roles_dir, d)
        if os.path.isdir(p) and (target is None or d == target):
            shutil.rmtree(p, ignore_errors=True)

    rows, list_lines, dropped_sil = [], [], 0.0
    for u in utts:
        name = u.speaker
        os.makedirs(os.path.join(roles_dir, name), exist_ok=True)
        seg = slice_audio(audio_e, sr_e, u.start, u.end, pad=0.0)
        if seg.size == 0:
            continue
        if args.strip_silence:
            before = len(seg)
            seg = strip_silence(seg, sr_e, min_gap=args.min_gap_silence)
            dropped_sil += (before - len(seg)) / float(sr_e)
        dur = len(seg) / float(sr_e)
        if dur < args.min_line_dur:
            continue
        fname = f"{name}_{u.index:04d}_{fmt_ts(u.start)}-{fmt_ts(u.end)}.wav"
        wav_path = os.path.join(roles_dir, name, fname)
        write_wav_mono(wav_path, seg, sr_e)
        with open(os.path.splitext(wav_path)[0] + ".txt", "w", encoding="utf-8") as f:
            f.write(u.text + "\n")
        rows.append({
            "index": u.index, "speaker": name,
            "start": round(u.start, 3), "end": round(u.end, 3),
            "dur": round(dur, 3), "text": u.text,
            "asr_conf": round(u.asr_conf, 4), "sv_conf": round(u.sv_conf, 4),
            "file": os.path.relpath(wav_path, out_dir).replace("\\", "/"),
        })
        list_lines.append(f"{os.path.abspath(wav_path)}|{name}|{lang}|{u.text}")

    if not rows:
        log("[x] 一条都没留下")
        return 5

    with open(os.path.join(out_dir, "lines.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "lines.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "gpt_sovits.list"), "w", encoding="utf-8") as f:
        f.write("\n".join(list_lines) + "\n")

    total = sum(float(r["dur"]) for r in rows)
    log("")
    log("=" * 58)
    log(f"完成，用时 {time.time() - t_all:.0f}s")
    log(f"  角色：{len({r['speaker'] for r in rows})}  台词：{len(rows)} 条 / {total:.1f}s")
    if args.strip_silence:
        log(f"  压掉静默：{dropped_sil:.1f}s")
    log(f"  目录：{roles_dir}")
    log(f"  清单：{os.path.join(out_dir, 'gpt_sovits.list')}")
    log("=" * 58)
    return 0


# ---------------------------------------------------------------- rename
def cmd_rename(args) -> int:
    run_dir = os.path.abspath(args.run_dir)
    with open(args.map, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    mapping = {str(k): safe_name(str(v)) for k, v in mapping.items()
               if str(v).strip() and str(k) != str(v)}

    roles = os.path.join(run_dir, "roles")
    if not os.path.isdir(roles):
        log(f"[x] 找不到 {roles}")
        return 2

    changed = 0
    for old, new in mapping.items():
        src = os.path.join(roles, old)
        dst = os.path.join(roles, new)
        if not os.path.isdir(src):
            log(f"[skip] 没有 {old}")
            continue
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.isdir(dst):
            log(f"[skip] 目标已存在：{new}")
            continue
        os.rename(src, dst)
        # 重命名内部文件前缀
        for fn in sorted(os.listdir(dst)):
            if fn.startswith(old + "_"):
                os.rename(os.path.join(dst, fn),
                          os.path.join(dst, fn[len(old) + 1:]))
        changed += 1
        log(f"  {old} -> {new}")

    # 重写 lines.csv / jsonl / list / speakers.csv
    lang = "zh"
    rc = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(rc):
        try:
            with open(rc, "r", encoding="utf-8") as f:
                rcj = json.load(f)
            lang = rcj.get("detected_language") or rcj.get("language") or "zh"
        except Exception:
            pass
    csv_path = os.path.join(run_dir, "lines.csv")
    rows = []
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    for r in rows:
        old = r["speaker"]
        if old in mapping:
            new = mapping[old]
            r["speaker"] = new
            r["file"] = r["file"].replace(f"roles/{old}/", f"roles/{new}/")
            # 文件名去前缀
            base = os.path.basename(r["file"])
            if base.startswith(old + "_"):
                r["file"] = os.path.join(os.path.dirname(r["file"]),
                                         base[len(old) + 1:]).replace("\\", "/")
    if rows:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(run_dir, "lines.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(run_dir, "gpt_sovits.list"), "w", encoding="utf-8") as f:
            for r in rows:
                abs_p = os.path.abspath(os.path.join(run_dir, r["file"]))
                f.write(f"{abs_p}|{r['speaker']}|{lang}|{r['text']}\n")
        # speakers.csv 跟着更新角色名
        with open(os.path.join(run_dir, "speakers.csv"), "w",
                  encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["speaker", "lines", "total_dur", "mean_sv_conf", "mean_asr_conf"])
            agg = {}
            for r in rows:
                a = agg.setdefault(r["speaker"], {"n": 0, "d": 0.0, "sv": 0.0, "asr": 0.0})
                a["n"] += 1
                a["d"] += float(r["dur"])
                a["sv"] += float(r["sv_conf"])
                a["asr"] += float(r["asr_conf"])
            for spk, a in sorted(agg.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
                w.writerow([spk, a["n"], round(a["d"], 2),
                            round(a["sv"] / a["n"], 4), round(a["asr"] / a["n"], 4)])

    log(f"改名完成，{changed} 个角色目录已更新。")
    return 0


# ---------------------------------------------------------------- fetch
def cmd_fetch(args) -> int:
    url = args.url
    out_dir = os.path.abspath(args.out or "assets")
    os.makedirs(out_dir, exist_ok=True)
    py = args.ytdlp_python or DEFAULT_YTDLP_PY
    ffmpeg_dir = os.path.dirname(find_ffmpeg(args.ffmpeg))
    env = dict(os.environ)
    env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    cmd = [py, "-m", "yt_dlp", "--no-warnings",
           "-f", args.format,
           "-o", os.path.join(out_dir, "%(id)s.%(ext)s")]
    if args.cookies_from_browser:
        cmd += ["--cookies-from-browser", args.cookies_from_browser]
    if args.cookies:
        cmd += ["--cookies", args.cookies]
    cmd.append(url)
    log("[fetch] " + " ".join(cmd[:6]) + f" ... {url}")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    p = subprocess.run(cmd, env=env, creationflags=flags)
    if p.returncode != 0:
        log("[x] 下载失败")
        return p.returncode
    files = [f for f in os.listdir(out_dir) if not f.endswith(".part")]
    log(f"[fetch] 完成，产物：{files}")
    return 0


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="anime_lines",
        description="番剧角色台词拆分：音轨 -> 声纹聚类 -> 转写 -> 按角色分目录")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="跑完整管线")
    r.add_argument("input", help="本地视频/音频文件")
    r.add_argument("-o", "--out-dir", required=True, help="输出目录")
    r.add_argument("--ffmpeg", default="")
    r.add_argument("--whisper-model", default="")
    r.add_argument("--campp-model", default="")
    r.add_argument("--language", default=None, help="ja/zh/en，默认自动检测")
    r.add_argument("--num-speakers", type=int, default=None, help="已知角色数时强制指定")
    r.add_argument("--cluster-threshold", type=float, default=0.55,
                   help="余弦距离阈值，越小角色越少（默认 0.55）")
    r.add_argument("--min-sv-dur", type=float, default=1.0, help="声纹可信最短时长")
    r.add_argument("--min-cluster-sec", type=float, default=3.0,
                   help="簇总时长低于此值则并进最相似的簇；设 0 关闭")
    r.add_argument("--min-line-dur", type=float, default=0.45)
    r.add_argument("--max-line-dur", type=float, default=20.0)
    r.add_argument("--merge-gap", type=float, default=0.35)
    r.add_argument("--split-gap", type=float, default=0.55)
    r.add_argument("--vad-pad", type=int, default=150)
    r.add_argument("--vad-threshold", type=float, default=0.35,
                   help="VAD 判定阈值，番剧 BGM 重时可降到 0.25~0.3 多捡语音")
    r.add_argument("--asr-device", default="cuda")
    r.add_argument("--asr-compute", default="int8_float16")
    r.add_argument("--device", default="cuda")
    r.add_argument("--export-sr", type=int, default=32000)
    r.add_argument("--keep", default="",
                   help="只处理这些区间，跳过 OP/ED/片尾废料。"
                        "写法：0-99,171-1324 或 0:00-1:39,2:51-22:04")
    r.add_argument("--drop-work", action="store_true", help="删掉 _work 中间件")
    r.set_defaults(func=cmd_run)

    n = sub.add_parser("rename", help="用映射表给角色起名并重组输出")
    n.add_argument("run_dir")
    n.add_argument("--map", required=True, help="speaker_map.json")
    n.set_defaults(func=cmd_rename)

    rf = sub.add_parser("refine",
                        help="按映射表合并多个簇、词级重判归属、清洗切片内的他人对话")
    rf.add_argument("run_dir", help="run 的输出目录（需保留 _work）")
    rf.add_argument("--map", required=True,
                    help='角色映射：{"久留美":["S01","S04"],"父亲":["S02"]} '
                         '或 {"S01":"久留美","S04":"久留美"}')
    rf.add_argument("--target", default=None,
                    help="只导出该角色（映射表里的角色名），不填则导出全部")
    rf.add_argument("-o", "--out-dir", default=None,
                    help="输出目录，默认 <run_dir>/clean_<角色>")
    rf.add_argument("--word-win", type=float, default=1.0,
                    help="滑窗声纹的窗长（秒），越短时间分辨率越高但越抖")
    rf.add_argument("--word-hop", type=float, default=0.25,
                    help="滑窗步长（秒）")
    rf.add_argument("--word-floor", type=float, default=0.30,
                    help="最高相似度低于此值视为非语音，丢弃该窗")
    rf.add_argument("--word-accept", type=float, default=0.52,
                    help="判定为某角色所需的绝对相似度；低于此值留空不硬分配")
    rf.add_argument("--word-margin", type=float, default=0.06,
                    help="（保留）切换所需的领先幅度")
    rf.add_argument("--word-fill-gap", type=float, default=0.70,
                    help="短于此秒数的空白若两侧同角色则回填")
    rf.add_argument("--word-min-run", type=float, default=0.75,
                    help="短于此秒数的异类段视为抖动，并回邻居")
    rf.add_argument("--edge-blur", type=float, default=0.50,
                    help="句首/句尾证据不足段短于此秒数则保留（避免吞字）")
    rf.add_argument("--strict", dest="strict", action="store_true", default=True,
                    help="严苛模式：句内沾到他人语音就整条丢弃（默认开）")
    rf.add_argument("--no-strict", dest="strict", action="store_false",
                    help="关闭严苛复核")
    rf.add_argument("--max-marks", type=int, default=1,
                    help="句内问号/感叹号超过此数量则判为多轮对话并丢弃")
    rf.add_argument("--near-sim", type=float, default=0.65,
                    help="与目标角色相似度 ≥ 此值的角色不视为「他人」（声线太近分不开）")
    rf.add_argument("--split-gap", type=float, default=0.55)
    rf.add_argument("--max-line-dur", type=float, default=20.0)
    rf.add_argument("--min-line-dur", type=float, default=0.45)
    rf.add_argument("--strip-silence", dest="strip_silence", action="store_true",
                    default=True, help="压掉切片内的长静默（默认开）")
    rf.add_argument("--no-strip-silence", dest="strip_silence",
                    action="store_false", help="保留原始静默")
    rf.add_argument("--min-gap-silence", type=float, default=0.35,
                    help="超过该时长的静默才压掉（默认 0.35s）")
    rf.add_argument("--export-sr", type=int, default=32000)
    rf.add_argument("--language", default=None, help="写进 list 的语言字段")
    rf.add_argument("--campp-model", default="")
    rf.add_argument("--device", default="cuda")
    rf.set_defaults(func=cmd_refine)

    f = sub.add_parser("fetch", help="用 yt-dlp 取音轨")
    f.add_argument("url")
    f.add_argument("-o", "--out", default="assets")
    f.add_argument("--format", default="bestaudio")
    f.add_argument("--ffmpeg", default="")
    f.add_argument("--ytdlp-python", default="")
    f.add_argument("--cookies", default="")
    f.add_argument("--cookies-from-browser", default="")
    f.set_defaults(func=cmd_fetch)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
