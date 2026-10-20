# -*- coding: utf-8 -*-
"""GPT-SoVITS v2 微调驱动：显式数据集、实验名和上游工程。

用法：
    python train_sovits.py --stage prep      # 三段预处理（文本/BERT、HuBERT、semantic）
    python train_sovits.py --stage sovits    # SoVITS 训练
    python train_sovits.py --stage gpt       # GPT 训练
    python train_sovits.py --stage all

设计要点：
- 采用 webui.py 的 open1abc / open1Ba / open1Bb 三条路径（环境变量传参、临时 config），
  不依赖 WebUI 界面，方便无人值守。
- 临时配置按实验名写到 TEMP，不改上游模板文件。质量与训练耗时须逐次实测。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import hashlib
import re
from pathlib import Path

ROOT = PY = EXP = EXP_DIR = TRAINSET = LIST = FLAT = TMP = ""

BERT = os.path.join(ROOT, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
HUBERT = os.path.join(ROOT, "GPT_SoVITS/pretrained_models/chinese-hubert-base")
V2 = os.path.join(ROOT, "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained")
S2G = os.path.join(V2, "s2G2333k.pth")
S2D = os.path.join(V2, "s2D2333k.pth")
S1 = os.path.join(V2, "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
S2CFG = os.path.join(ROOT, "GPT_SoVITS/configs/s2.json")

SOVITS_OUT = os.path.join(ROOT, "SoVITS_weights_v2")
GPT_OUT = os.path.join(ROOT, "GPT_weights_v2")

# 训练超参（8GB 显存 / 8 分钟数据）
S2_BATCH, S2_EPOCHS, S2_SAVE_EVERY = int(os.environ.get("GSV_S2_BATCH", 6)), int(os.environ.get("GSV_S2_EPOCHS", 8)), 4
S1_BATCH, S1_EPOCHS, S1_SAVE_EVERY = int(os.environ.get("GSV_S1_BATCH", 4)), int(os.environ.get("GSV_S1_EPOCHS", 15)), 5


def log(msg):
    print(msg, flush=True)


def run(script, env_extra, tag, extra=None):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_extra.items()})
    cmd = [PY, "-s", script] + list(extra or [])
    log(f"[{tag}] {' '.join(os.path.basename(c) for c in cmd)}")
    p = subprocess.run(cmd, cwd=ROOT, env=env)
    if p.returncode != 0:
        raise SystemExit(f"[{tag}] 失败，返回码 {p.returncode}")


def dataset():
    src = Path(TRAINSET) / "train.list"
    rows = []
    digest = hashlib.sha256()
    for i, line in enumerate(src.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip(): continue
        parts = line.split("|", 3)
        if len(parts) != 4 or any(not x.strip() for x in parts) or '|' in parts[-1]:
            raise SystemExit(f"Invalid train.list row {i}")
        path, spk, lang, text = parts
        audio = Path(path)
        if not audio.is_absolute() or not audio.is_file():
            raise SystemExit(f"Row {i}: expected existing absolute audio path: {path}")
        content_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
        digest.update((str(audio.resolve()) + "|" + content_hash + "|" + spk + "|" + lang + "|" + text + "\n").encode("utf8"))
        rows.append((audio,spk,lang,text,content_hash))
    if not rows: raise SystemExit("Empty train.list")
    return rows, digest.hexdigest()


def ensure_list():
    rows, fingerprint = dataset()
    names = [p.name.casefold() for p,_,_,_,_ in rows]
    need_flat = len(set(names)) != len(names)
    out_rows = []
    for index,(audio,spk,lang,text,content_hash) in enumerate(rows):
        if need_flat:
            # Index + path + content prevent cross-episode and stale-copy collisions.
            suffix = hashlib.sha256((str(audio.resolve())+content_hash).encode("utf8")).hexdigest()[:16]
            dst = Path(FLAT) / f"{index:06d}_{suffix}{audio.suffix.lower()}"
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists(): shutil.copy2(audio,dst)
        else: dst = audio.resolve()
        out_rows.append(f"{dst.as_posix()}|{spk}|{lang}|{text}")
    Path(LIST).write_text("\n".join(out_rows)+"\n",encoding="utf8")
    log(f"[list] {len(out_rows)} rows; sha256={fingerprint}; output={LIST}")
    return len(out_rows)


def verify_experiment(create=False):
    rows,fingerprint=dataset()
    manifest=Path(EXP_DIR)/"dataset-manifest.json"
    if manifest.exists():
        recorded=json.loads(manifest.read_text(encoding="utf8"))
        if recorded["sha256"] != fingerprint:
            raise SystemExit("Dataset changed: choose a NEW --exp; cached features cannot be reused")
    elif Path(EXP_DIR).exists() and any(Path(EXP_DIR).iterdir()):
        raise SystemExit("Existing experiment has no dataset manifest; use a NEW --exp")
    elif create:
        manifest.parent.mkdir(parents=True,exist_ok=True)
        manifest.write_text(json.dumps(dict(sha256=fingerprint,rows=len(rows),trainset=TRAINSET),indent=2),encoding="utf8")
    else:
        raise SystemExit("Run --stage prep first in a fresh experiment")
    return len(rows)


def verify_features():
    n=verify_experiment()
    expected={line.split("|",1)[0].replace("\\","/").rsplit("/",1)[-1] for line in Path(LIST).read_text(encoding="utf8").splitlines() if line.strip()}
    for file,header in [("2-name2text.txt",False),("6-name2semantic.tsv",True)]:
        lines=(Path(EXP_DIR)/file).read_text(encoding="utf8").splitlines()
        if header: lines=lines[1:]
        keys=[line.split("\t",1)[0] for line in lines if line.strip()]
        if len(keys)!=n or set(keys)!=expected:
            raise SystemExit(f"Incomplete/colliding features: {file}: {len(keys)} / {n}")
    for folder in ["4-cnhubert","5-wav32k"]:
        files=list((Path(EXP_DIR)/folder).iterdir())
        keys={f.name.removesuffix(".pt") for f in files} if folder=="4-cnhubert" else {f.name for f in files}
        if not expected.issubset(keys):
            raise SystemExit(f"Incomplete features: {folder}")


def stage_prep():
    verify_experiment(create=True)
    n = ensure_list()
    os.makedirs(EXP_DIR, exist_ok=True)
    base = dict(
        inp_text=LIST, inp_wav_dir="", exp_name=EXP, opt_dir=EXP_DIR,
        i_part="0", all_parts="1", _CUDA_VISIBLE_DEVICES="0",
        is_half="True", version="v2",
    )

    # 1A 文本 → BERT 特征 → 2-name2text.txt
    p2 = os.path.join(EXP_DIR, "2-name2text.txt")
    if not (os.path.exists(p2) and os.path.getsize(p2) > 10):
        run("GPT_SoVITS/prepare_datasets/1-get-text.py",
            dict(base, bert_pretrained_dir=BERT), "1A 文本+BERT")
        # 清理分片
        for f in os.listdir(EXP_DIR):
            if f.startswith("2-name2text-") and f != "2-name2text.txt":
                rows = open(os.path.join(EXP_DIR, f), encoding="utf-8").read().strip("\n").split("\n")
                with open(p2, "a", encoding="utf-8") as w:
                    w.write("\n".join(rows) + "\n")
                os.remove(os.path.join(EXP_DIR, f))
    log(f"[1A] {sum(1 for _ in open(p2, encoding='utf-8'))} 行")

    # 1B 32k 重采样 + HuBERT 特征
    p5 = os.path.join(EXP_DIR, "5-wav32k")
    if not (os.path.isdir(p5) and len(os.listdir(p5)) >= n * 0.98):
        run("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
            dict(base, cnhubert_base_dir=HUBERT), "1B HuBERT")
    log(f"[1B] wav32k {len(os.listdir(p5))} / {n}，cnhubert {len(os.listdir(os.path.join(EXP_DIR,'4-cnhubert')))}")

    # 1C semantic token
    p6 = os.path.join(EXP_DIR, "6-name2semantic.tsv")
    if not (os.path.exists(p6) and os.path.getsize(p6) > 31):
        run("GPT_SoVITS/prepare_datasets/3-get-semantic.py",
            dict(base, pretrained_s2G=S2G, s2config_path=S2CFG), "1C semantic")
        # 分片合并（与 webui open1abc 的 1C 收尾一致：表头 + 各分片）
        opt = ["item_name\tsemantic_audio"]
        for f in sorted(os.listdir(EXP_DIR)):
            if f.startswith("6-name2semantic-") and f.endswith(".tsv"):
                fp = os.path.join(EXP_DIR, f)
                opt += open(fp, encoding="utf-8").read().strip("\n").split("\n")
                os.remove(fp)
        with open(p6, "w", encoding="utf-8") as w:
            w.write("\n".join(opt) + "\n")
    log(f"[1C] semantic {sum(1 for _ in open(p6, encoding='utf-8')) - 1} 行")


def stage_sovits():
    verify_features()
    # s2_train 会往 <exp_dir>/logs_s2_<version>/ 写 checkpoint，webui 里由 open1Ba 预建；
    # 这里必须自己建，否则 shutil.move 报 WinError 3（父目录不存在）。
    os.makedirs(os.path.join(EXP_DIR, "logs_s2_v2"), exist_ok=True)
    os.makedirs(SOVITS_OUT, exist_ok=True)
    cfg = json.load(open(S2CFG, encoding="utf-8"))
    cfg["train"].update(
        batch_size=S2_BATCH, epochs=S2_EPOCHS, text_low_lr_rate=0.4,
        pretrained_s2G=S2G, pretrained_s2D=S2D,
        if_save_latest=True, if_save_every_weights=True,
        save_every_epoch=S2_SAVE_EVERY, gpu_numbers="0",
        grad_ckpt=True, lora_rank=32,
    )
    cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = EXP_DIR
    cfg["save_weight_dir"] = SOVITS_OUT
    cfg["name"] = EXP
    cfg["version"] = "v2"
    cfg["model"]["version"] = "v2"
    tmp = os.path.join(TMP, f"_{EXP}_s2.json")
    json.dump(cfg, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    log(f"[S2] batch={S2_BATCH} epochs={S2_EPOCHS} grad_ckpt=True → {SOVITS_OUT}")
    run("GPT_SoVITS/s2_train.py", dict(_CUDA_VISIBLE_DEVICES="0", version="v2"),
        "S2 训练", extra=["--config", tmp])


def stage_gpt():
    verify_features()
    import yaml
    cfg = yaml.load(open(os.path.join(ROOT, "GPT_SoVITS/configs/s1longer-v2.yaml"),
                         encoding="utf-8"), Loader=yaml.FullLoader)
    cfg["train"].update(
        batch_size=S1_BATCH, epochs=S1_EPOCHS,
        save_every_n_epoch=S1_SAVE_EVERY, if_save_every_weights=True,
        if_save_latest=True, if_dpo=False,
        half_weights_save_dir=GPT_OUT, exp_name=EXP,
    )
    cfg["pretrained_s1"] = S1
    cfg["train_semantic_path"] = os.path.join(EXP_DIR, "6-name2semantic.tsv")
    cfg["train_phoneme_path"] = os.path.join(EXP_DIR, "2-name2text.txt")
    cfg["output_dir"] = os.path.join(EXP_DIR, "logs_s1_v2")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    tmp = os.path.join(TMP, f"_{EXP}_s1.yaml")
    yaml.dump(cfg, open(tmp, "w", encoding="utf-8"), default_flow_style=False, allow_unicode=True)
    log(f"[S1] batch={S1_BATCH} epochs={S1_EPOCHS} → {GPT_OUT}")
    run("GPT_SoVITS/s1_train.py", dict(_CUDA_VISIBLE_DEVICES="0", hz="25hz"),
        "S1 训练", extra=["--config_file", tmp])


def report():
    log("=" * 56)
    for d in (SOVITS_OUT, GPT_OUT):
        if not os.path.isdir(d):
            continue
        hits = sorted(f for f in os.listdir(d) if EXP in f)
        log(f"{os.path.basename(d)}: {hits if hits else '（无）'}")
    sem = os.path.join(EXP_DIR, "6-name2semantic.tsv")
    if os.path.exists(sem):
        log(f"semantic 行数 {sum(1 for _ in open(sem, encoding='utf-8')) - 1}")


def main():
    global ROOT,PY,EXP,EXP_DIR,TRAINSET,LIST,FLAT,TMP,BERT,HUBERT,V2,S2G,S2D,S1,S2CFG,SOVITS_OUT,GPT_OUT
    ap=argparse.ArgumentParser(description="GPT-SoVITS v2 training; explicit data and experiment required")
    ap.add_argument("--root",default=os.environ.get("GSV_ROOT"))
    ap.add_argument("--python",default=os.environ.get("GSV_PYTHON",sys.executable))
    ap.add_argument("--exp",default=os.environ.get("GSV_EXP"))
    ap.add_argument("--trainset",default=os.environ.get("GSV_TRAINSET"))
    ap.add_argument("--stage",default="all",choices=["check","list","prep","sovits","gpt","all"])
    a=ap.parse_args()
    if not a.exp or not re.fullmatch(r"[A-Za-z0-9_-]+",a.exp): ap.error("explicit safe --exp/GSV_EXP is required")
    if not a.trainset: ap.error("explicit --trainset/GSV_TRAINSET is required")
    if not a.root: ap.error("--root/GSV_ROOT is required")
    ROOT=str(Path(a.root).resolve()); PY=str(Path(a.python).resolve()); EXP=a.exp
    EXP_DIR=str(Path(ROOT)/"logs"/EXP); TRAINSET=str(Path(a.trainset).resolve())
    LIST=str(Path(TRAINSET)/"gw_list.txt"); FLAT=str(Path(TRAINSET)/"gpt"); TMP=str(Path(ROOT)/"TEMP")
    BERT=str(Path(ROOT)/"GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
    HUBERT=str(Path(ROOT)/"GPT_SoVITS/pretrained_models/chinese-hubert-base")
    V2=str(Path(ROOT)/"GPT_SoVITS/pretrained_models/gsv-v2final-pretrained")
    S2G=str(Path(V2)/"s2G2333k.pth"); S2D=str(Path(V2)/"s2D2333k.pth")
    S1=str(Path(V2)/"s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
    S2CFG=str(Path(ROOT)/"GPT_SoVITS/configs/s2.json")
    SOVITS_OUT=str(Path(ROOT)/"SoVITS_weights_v2"); GPT_OUT=str(Path(ROOT)/"GPT_weights_v2")
    rows,fingerprint=dataset()
    log(json.dumps(dict(stage=a.stage,experiment=EXP,trainset=TRAINSET,rows=len(rows),sha256=fingerprint),ensure_ascii=False))
    if a.stage == "check": return
    if a.stage == "list": ensure_list(); return
    os.makedirs(TMP,exist_ok=True)
    if a.stage in ["prep","all"]: stage_prep(); verify_features()
    if a.stage in ["sovits","all"]: stage_sovits()
    if a.stage in ["gpt","all"]: stage_gpt()
    report()

if __name__ == "__main__":
    main()
