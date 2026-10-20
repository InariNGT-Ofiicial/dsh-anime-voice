#!/usr/bin/env python
"""Run explicit, restartable stages. A dry run never proves execution."""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["probe", "raw", "sep", "asr", "slice", "spk", "report", "train"]


def parse_keep(items):
    keep = {}
    for item in items:
        ep, spec = item.split("=", 1)
        spans = sorted(tuple(map(float, span.split("-", 1))) for span in spec.split(","))
        if not spans or any(a < 0 or b <= a for a, b in spans):
            raise ValueError("keep spans require 0 <= start < end")
        if any(spans[i][1] > spans[i+1][0] for i in range(len(spans)-1)):
            raise ValueError("keep spans must not overlap")
        keep[ep] = spans
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--eps", default="ep01")
    ap.add_argument("--role", default="")
    ap.add_argument("--lang", choices=["ja", "zh", "en"], default="ja")
    ap.add_argument("--keep", action="append", default=[])
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--old", type=Path)
    ap.add_argument("--spk", type=float, default=0.55)
    ap.add_argument("--f0-min", type=float, default=0, help="Optional pitch filter; tune for the target, 0 disables it")
    ap.add_argument("--stages", default="probe,raw,sep,asr,slice,spk")
    ap.add_argument("--from", dest="start", default="")
    ap.add_argument("--python", default=os.environ.get("ANIME_PYTHON", sys.executable))
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG_BIN", "ffmpeg"))
    ap.add_argument("--extractor", type=Path, default=Path(os.environ.get("ANIME_EXTRACTOR_SCRIPTS", HERE.parents[1] / "anime-voice-line-extractor/scripts")))
    ap.add_argument("--sep-root", type=Path, default=os.environ.get("ANIME_SEP_ROOT"))
    ap.add_argument("--sep-python", default=os.environ.get("ANIME_SEP_PYTHON"))
    ap.add_argument("--sep-config", default=os.environ.get("ANIME_SEP_CONFIG"))
    ap.add_argument("--sep-checkpoint", default=os.environ.get("ANIME_SEP_CHECKPOINT"))
    ap.add_argument("--exp", default=os.environ.get("GSV_EXP"))
    ap.add_argument("--trainset", type=Path, default=os.environ.get("GSV_TRAINSET"))
    ap.add_argument("--train-stage", choices=["list", "prep", "sovits", "gpt", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-stages", action="store_true")
    a = ap.parse_args()
    if a.list_stages:
        print(", ".join(STAGES)); return 0
    if not a.work:
        ap.error("--work is required")
    work = a.work.resolve()
    eps = [x.strip() for x in a.eps.split(",") if x.strip()]
    if not eps or any(not re.fullmatch(r"[A-Za-z0-9_-]+", ep) for ep in eps):
        ap.error("episode IDs must be nonempty safe file stems")
    chosen = STAGES[:] if a.stages == "all" else a.stages.split(",")
    if not chosen or any(s not in STAGES for s in chosen):
        ap.error("unknown/empty stage selection")
    if a.start:
        if a.start not in chosen: ap.error("--from is not in selected stages")
        chosen = chosen[chosen.index(a.start):]
    try: keep = parse_keep(a.keep)
    except ValueError as e: ap.error(str(e))
    if any(s in chosen for s in ["raw", "slice"]):
        if any(ep not in keep for ep in eps): ap.error("explicit --keep ep=start-end is required for raw/slice")
    if any(s in chosen for s in ["slice", "spk"]) and not a.role:
        ap.error("--role is required for slice/spk")
    if "spk" in chosen and not a.ref: ap.error("--ref is required for spk")
    if "sep" in chosen and not all([a.sep_root,a.sep_python,a.sep_config,a.sep_checkpoint]):
        ap.error("sep requires --sep-root/--sep-python/--sep-config/--sep-checkpoint")
    if "train" in chosen and not a.exp: ap.error("train requires explicit --exp/GSV_EXP")
    full = work / "_full"
    trainset = (a.trainset or full / "out/ts").resolve()
    def py(script,*args): return [a.python,str(script),*map(str,args)]
    jobs = []
    def add(stage,cwd,argv): jobs.append((stage,cwd,argv))
    for stage in chosen:
        if stage in ["probe","raw"]:
            for ep in eps:
                src = work / "src" / (ep+".m4a")
                if stage == "probe":
                    add(stage,work,py(a.extractor / "profile_episode.py",src,"--block",30))
                    add(stage,work,py(a.extractor / "stereo_probe.py",src,"--start",80,"--end",200,"--step",2))
                else:
                    add(stage,work,[a.ffmpeg,"-y","-i",str(src),"-vn","-ac","2","-ar","44100","-t",str(max(b for _,b in keep[ep])),str(full/"raw"/(ep+".wav"))])
        elif stage == "sep":
            # Stage only the requested episodes, avoiding unrelated stale raw files.
            add(stage,a.sep_root,[a.sep_python,"inference.py","--model_type","bs_roformer","--config_path",a.sep_config,"--start_check_point",a.sep_checkpoint,"--input_folder",str(full/"sep_input"),"--store_dir",str(full/"voc"),"--disable_detailed_pbar"])
        elif stage == "asr":
            for script in ["asr_full.py","asr_full2.py"]:
                add(stage,work,py(HERE/script,"--base",work,"--eps",",".join(eps),"--lang",a.lang))
        elif stage == "slice":
            spec = ";".join(ep+"="+",".join(f"{x}-{y}" for x,y in keep[ep]) for ep in eps)
            add(stage,work,py(HERE/"split_full.py","--base",work,"--eps",",".join(eps),"--keep",spec,"--role",a.role,"--lang",a.lang,"--plan","--slice"))
        elif stage == "spk":
            add(stage,work,py(HERE/"filter_speaker.py","--base",work,"--ref",a.ref.resolve(),"--role",a.role,"--lang",a.lang,"--spk",a.spk,"--f0-min",a.f0_min))
        elif stage == "report":
            argv=py(HERE/"make_dataset_report.py","--new",full/"out/ts/train.list","--out",work/"out/dataset-report.html")
            if a.old: argv += ["--old",str(a.old.resolve())]
            add(stage,work,argv)
        elif stage == "train":
            add(stage,work,py(HERE/"train_sovits.py","--stage",a.train_stage,"--exp",a.exp,"--trainset",trainset))
    if a.dry_run:
        for stage,cwd,argv in jobs: print(f"[{stage}] cwd={cwd} " + subprocess.list2cmdline(argv))
        print("[dry-run] no commands executed; dependencies not verified")
        return 0
    if not work.is_dir(): ap.error("work directory does not exist")
    env = os.environ.copy()
    env["ANIME_EXTRACTOR_SCRIPTS"] = str(a.extractor.resolve())
    records=[]
    status=work/"pipeline-status.json"
    if "raw" in chosen: (full/"raw").mkdir(parents=True,exist_ok=True)
    if "sep" in chosen:
        import shutil
        staging=full/"sep_input"
        if staging.exists() and any(staging.iterdir()):
            ap.error("sep_input must be empty; use a fresh work directory or explicitly clear reviewed staging files")
        staging.mkdir(parents=True,exist_ok=True)
        for ep in eps: shutil.copy2(full/"raw"/(ep+".wav"),staging/(ep+".wav"))
        (full/"voc").mkdir(parents=True,exist_ok=True)
    for stage,cwd,argv in jobs:
        record=dict(stage=stage,argv=argv,cwd=str(cwd),started=time.time(),state="running")
        records.append(record)
        status.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf8")
        print(f"[{stage}] " + subprocess.list2cmdline(argv),flush=True)
        try:
            result=subprocess.run(argv,cwd=cwd,env=env)
            record.update(state="passed" if result.returncode==0 else "failed",returncode=result.returncode,finished=time.time())
        except OSError as e:
            record.update(state="failed",error=str(e),finished=time.time())
        status.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf8")
        if record["state"] != "passed": raise SystemExit(f"Stage {stage} failed; see {status}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
