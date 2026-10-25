#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
env_probe.py —— 一次性把 GPT-SoVITS 训练环境的实况吐成 JSON。

为什么要一个脚本搞定全部：
    每次 spawn 一个 Python 要 0.3~1s，逐项探测会拖到十几秒；
    而且 import torch 失败的进程会污染后续 import，一次一个进程最干净。
    **每个 import 都单独 try**：漏一个就看不出"到底缺什么"。
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys


def mod(name):
    try:
        m = __import__(name)
        return getattr(m, "__version__", "ok"), None
    except Exception as e:                                    # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def probe_torch():
    out = {"installed": False}
    try:
        import torch
        out["installed"] = True
        out["version"] = torch.__version__
        out["cudaBuild"] = torch.version.cuda
        out["cudaAvailable"] = bool(torch.cuda.is_available())
        if out["cudaAvailable"]:
            out["deviceName"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            out["capability"] = f"sm_{cap[0]}{cap[1]}"
            out["totalVramGB"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
            try:
                used = torch.cuda.memory_allocated(0) / 1024 ** 3
                out["allocatedGB"] = round(used, 2)
            except Exception:
                pass
    except Exception as e:                                    # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def probe_ffmpeg(candidate):
    for exe in [candidate, "ffmpeg"]:
        if not exe:
            continue
        p = exe if os.path.isabs(exe) else shutil.which(exe)
        if not p or not os.path.isfile(p):
            continue
        try:
            r = subprocess.run([p, "-version"], capture_output=True, timeout=15)
            first = (r.stdout or b"").decode("utf8", "replace").splitlines()
            return {"path": os.path.abspath(p), "version": (first[0] if first else "")[:80]}
        except Exception as e:                                # noqa: BLE001
            return {"path": os.path.abspath(p), "error": str(e)}
    return {}


def probe_asr_cache(cache_root, model):
    if not cache_root:
        return {"found": False, "reason": "未指定缓存目录"}
    hub = os.path.join(cache_root, "hub")
    d = os.path.join(hub, f"models--Systran--faster-whisper-{model}", "snapshots")
    if not os.path.isdir(d):
        return {"found": False, "cacheRoot": cache_root, "hub": hub}
    revs = [os.path.join(d, x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
    for r in revs:
        mb = os.path.join(r, "model.bin")
        if os.path.isfile(mb):
            return {"found": True, "path": r, "modelBinMB": round(os.path.getsize(mb) / 1024 ** 2, 1)}
    return {"found": False, "cacheRoot": cache_root, "reason": "快照不完整（缺 model.bin）"}


def main():
    args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    gsv = args.get("gsvDir") or ""
    asr_model = args.get("asrModel") or "large-v3"
    cache = args.get("asrCacheDir") or os.environ.get("HF_HOME") or ""
    ffmpeg = args.get("ffmpeg") or ""
    deep = bool(args.get("deep"))

    out = {
        "python": {
            "exe": sys.executable,
            "version": platform.python_version(),
            "bits": platform.architecture()[0],
            "prefixVenv": sys.prefix != sys.base_prefix,
        },
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "torch": probe_torch(),
        "ffmpeg": probe_ffmpeg(ffmpeg),
        "asrCache": probe_asr_cache(cache, asr_model),
        "modules": {},
        "modulesSkipped": [],
        "deep": deep,
        "disk": {},
    }

    # ★ 分两档，因为「装了什么」和「能不能训」是两件事，不必都付时间成本：
    #   轻档 = 参与判定的模块（yaml 缺了就出不了 s1 配置；faster_whisper 只影响 ASR 兜底）
    #   重档 = 纯信息性，实测 peft 6.7s + transformers 4.4s + gradio 3.0s，为体检白等 14 秒
    LIGHT = ("faster_whisper", "yaml", "soundfile", "numpy", "ctranslate2", "onnxruntime")
    HEAVY = ("librosa", "transformers", "gradio", "peft", "flash_attn")
    for name in LIGHT:
        v, err = mod(name)
        out["modules"][name] = {"version": v} if v else {"error": err}
    if deep:
        for name in HEAVY:
            v, err = mod(name)
            out["modules"][name] = {"version": v} if v else {"error": err}
    else:
        out["modulesSkipped"] = list(HEAVY)

    # GPT-SoVITS 必须先 chdir 才能 import（它用相对路径读 config）
    if gsv and os.path.isdir(gsv):
        os.chdir(gsv)
        sys.path.insert(0, gsv)
        sys.path.insert(0, os.path.join(gsv, "GPT_SoVITS"))
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(k, None)
        for name in ("TTS_infer_pack.TTS", "text.cleaner"):
            try:
                __import__(name)
                out["modules"][name] = {"import": "ok"}
            except Exception as e:                            # noqa: BLE001
                out["modules"][name] = {"error": f"{type(e).__name__}: {e}"}

    for label, p in (("gsv", gsv),):
        try:
            if p and os.path.isdir(p):
                u = shutil.disk_usage(p)
                out["disk"][label] = {"freeGB": round(u.free / 1024 ** 3, 1),
                                      "totalGB": round(u.total / 1024 ** 3, 1)}
        except Exception:
            pass

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
