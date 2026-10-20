---
name: anime-voice-clone-training
description: 从整集番剧正片分离干声、带标点 ASR、标点切句和声纹筛选构建 GPT-SoVITS v2 数据集，显式选择实验与数据集训练，并用已有检查点短句配音验证。仅抽台词使用 anime-voice-line-extractor；不用于歌曲换声。
---

# GPT-SoVITS v2 训练与配音

脚本在本技能 `scripts/`。使用已配置的 GPT-SoVITS v2 Python；`GSV_ROOT` 指向上游工程，`GSV_PYTHON` 指向它的解释器。不要重装或与其它声音项目混装。与歌曲 DDSP 训练插件是不同项目。

## 数据路线

A 建立人工确认的角色参照。B 主路线是整集正片 → 整集 MSST 分离 → `asr_full.py` 整条对照 ASR → `asr_full2.py` 分块重注标点 ASR → `split_full.py` 标点切句/交叉校验 → `filter_speaker.py` 声纹筛选 → 人工校对 → 训练。

用 `run_pipeline.py --list-stages` 查看阶段。输入布局 `WORK/src/ep01.m4a`；必须提供每集明确的 `--keep ep01=0-900`。输出 `_full/raw/`、`_full/voc/<ep>/vocals.wav`、`_full/asr/`、`_full/out/wavs/` 与 `_full/out/ts/train.list`。可以 `--stages report` 只复用现有清单做报告，或用 `--from` 继续。`pipeline-status.json` 记录每次真实命令和状态；`--dry-run` 只证明命令规划。

声纹阶段必须指定 `--ref` 已确认的同人 WAV 目录。默认声纹阈值 0.55 是初始筛选值；F0 下限默认 0，可按声音特点显式调整。角色判断、标点和音频收尾都需要试听，不能用单个分数代替人工审核。

## 训练接口

必须显式设置 `GSV_EXP` 和 `GSV_TRAINSET`（或 `--exp` / `--trainset`）。没有旧项目的默认实验或数据路径。清单格式为 `绝对wav路径|角色|语言|文本`。

```powershell
& $env:GSV_PYTHON "$S/train_sovits.py" --stage check --exp character_v1 --trainset "$W/_full/out/ts"
& $env:GSV_PYTHON "$S/train_sovits.py" --stage list --exp character_v1 --trainset "$W/_full/out/ts"
# 仅在用户授权训练、已人工校对且 GPU 空闲后：
& $env:GSV_PYTHON "$S/train_sovits.py" --stage prep --exp character_v1 --trainset "$W/_full/out/ts"
& $env:GSV_PYTHON "$S/train_sovits.py" --stage sovits --exp character_v1 --trainset "$W/_full/out/ts"
& $env:GSV_PYTHON "$S/train_sovits.py" --stage gpt --exp character_v1 --trainset "$W/_full/out/ts"
```

`check` 只读校验、统计并打印数据指纹；`list` 只生成 `gw_list.txt`，重名时生成唯一音频名，不调用模型。`prep` 才提取文本/BERT、HuBERT、semantic 特征。数据路径、文本或音频变化会使实验指纹不符，必须使用新实验名。已有旧实验没有指纹时拒绝复用，避免将旧特征冒充新训练数据。

`run_pipeline.py --stages train --exp character_v1 --trainset ... --train-stage list` 会真实调用驱动。`--train-stage all` 会启动训练，不能在未授权的验证任务中使用。

历史本地案例旧集 157 条与整集重建 291 条是两套数据，不自动继承。验证当前案例时应显式选择对应的 `kurumi_v3` 实验和新集目录，不能从旧默认值猜测。

## 使用已有权重短配音

`infer_tts.py --root ROOT --sovits MODEL.pth --gpt MODEL.ckpt --ref REF.wav --prompt-text "参考原文" --text "新句子" --lang ja --out NEW.wav`。

所有模型必须显式选择，参考音频 3–10 秒。输出原始 WAV 和同名 JSON（权重/参考/输出 SHA-256、采样率、种子、时长）。不覆盖已有输出。保留生成器原有气口与尾音，不自动裁剪或拼接。先检查 GPU 是否空闲；不能停止其它任务。

生成成功仅证明技术链路运行；还需逐句 ASR 和人工复听检查漏字、尾字、语种、音色。不能宣称“不可区分”“多少分钟稳定可用”或复用旧模型质量数字。不发布声音素材、模型、会话或凭据。
