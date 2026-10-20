# 本地配置与复现

## 依赖和版本

宿主技能使用 `SKILL.md` 加脚本，不是修改 DSH 内核的二进制插件。两个技能应装在同一技能根；B 默认从相邻目录导入 A，也可用 `ANIME_EXTRACTOR_SCRIPTS` 指向 A 的 `scripts`。

分析 Python 需要 numpy、scipy、soundfile、faster-whisper、funasr、modelscope、scikit-learn；F0 诊断需要 pyworld。GPU 相关 torch、torchaudio、CUDA 库必须匹配现有环境。`requirements-analysis.txt` 是依赖范围声明，完整已测版本在验证记录中登记；不保证任意版本组合都兼容。

模型不随仓库下载：

- faster-whisper large-v3：`ANIME_WHISPER_MODEL` 指向含 `model.bin` 的目录。
- cam++：`ANIME_CAMPP_MODEL` 指向含 `campplus_cn_common.bin` 的目录或 snapshots 目录。
- speech BS-RoFormer：通过 `ANIME_SEP_CONFIG` / `ANIME_SEP_CHECKPOINT` 指定兼容 MSST 的配置和检查点，路径相对于 `ANIME_SEP_ROOT`。
- GPT-SoVITS v2：上游 BERT、HuBERT、v2 预训练权重和 s1/s2 配置必须完整。推理也需要上游工程，不能只给两个微调权重。

`config.example.ps1` 中路径都是占位示例。填成自己的路径后 dot-source，每个新的 PowerShell/DSH 工具进程需设置相同环境变量。不要把本地配置提交到仓库。

## A：台词和训练参照

```powershell
. ./config.local.ps1
$A = "$env:USERPROFILE/.dsh/skills/anime-voice-line-extractor/scripts"
$B = "$env:USERPROFILE/.dsh/skills/anime-voice-clone-training/scripts"
$W = 'E:/Projects/my_character'
& $env:ANIME_PYTHON "$A/profile_episode.py" "$W/src/ep01.m4a" --block 30
# 以下区间只用于示例，必须换成人工确认的本片区间。
& $env:ANIME_PYTHON "$A/anime_lines.py" run "$W/src/ep01.m4a" -o "$W/lines/ep01" --language ja --keep '0:00-1:30,2:40-22:00'
# roles.json 例如 {"目标角色":["S01","S03"],"其它角色":["S02"]}
& $env:ANIME_PYTHON "$A/anime_lines.py" refine "$W/lines/ep01" --map "$W/roles.json" --target '目标角色' -o "$W/refined/ep01"
& $env:ANIME_PYTHON "$A/export_trainset.py" "$W/refined/ep01" --src "$W/lines/ep01" --out "$W/exports/ep01" --min-dur 1.5 --lang ja
```

`train.list`、WAV 和 `待校对.tsv` 在输出目录。绝对音频路径只存在本地生成清单中。多集分别导出，再用 `merge_trainset.py PARENT --order ep01,ep02` 合并。`gen_preview.py REFINED` 生成台词试听页。

声音相近、重叠对白、喊叫、音乐和 ASR 同音字都可能造成误判。先人工确认参照，不能把角色簇号或相似度直接当角色真值。

## B：整集重建训练集

```powershell
& $env:ANIME_PYTHON "$B/run_pipeline.py" --work $W --eps ep01 --role '目标角色' --lang ja --keep 'ep01=0-90,160-1320' --ref "$W/ref" --dry-run
& $env:ANIME_PYTHON "$B/run_pipeline.py" --work $W --eps ep01 --role '目标角色' --lang ja --keep 'ep01=0-90,160-1320' --ref "$W/ref" --stages raw,sep,asr,slice,spk,report
```

`raw` 保留从 0 开始的源时间轴，只截到最后一个保留区间末尾；OP/ED 在切句时剔除，因此各阶段时间戳可回查源片。ASR 从 raw 混流转写；切片从分离后的连续干声提取。主 ASR 与整条对照版都必须存在。

`sep` 仅暂存所选集，避免把其它集的旧文件一起分离。运行后保留 `_full/sep_input`；再次分离前须人工检查并清空该暂存目录，或用新工作区。分离器应输出 `_full/voc/<ep>/vocals.wav`；其它 MSST 分支的命名可能不同，需要按实际输出适配。

`run_pipeline.py` 的阶段顺序按 `--stages` 保留，`--from` 从所选阶段继续。真实执行写 `pipeline-status.json`；非零退出或子进程启动失败会立即停止。模型权重、源音频或 ASR 配置改变时请用新工作区，或显式重跑相关阶段；旧 ASR 文件的存在检查不能证明它仍与当前输入相符。

## 显式训练，避免串用旧数据

```powershell
$env:GSV_EXP = 'my_character_v1'
$env:GSV_TRAINSET = "$W/_full/out/ts"
& $env:GSV_PYTHON "$B/train_sovits.py" --stage check
& $env:GSV_PYTHON "$B/train_sovits.py" --stage list
# 人工校对且允许训练后，才执行下面这些 GPU 阶段。
& $env:GSV_PYTHON "$B/train_sovits.py" --stage prep
& $env:GSV_PYTHON "$B/train_sovits.py" --stage sovits
& $env:GSV_PYTHON "$B/train_sovits.py" --stage gpt
```

`check` 输出实际清单条数和 SHA-256 数据指纹，不创建实验。`list` 创建 `gw_list.txt`，重名音频扁平化到 `gpt/`；绝不调用预处理。`prep` 在上游 `logs/EXP/dataset-manifest.json` 记录数据指纹，并核对所有特征键完整匹配；训练阶段拒绝不匹配或没有指纹的旧实验。源音频、路径或文本有变化时使用新实验名。已有历史实验不做自动迁移。

流水线的 `--stages train --exp EXP --trainset DIR --train-stage list` 同样真实执行清单生成；默认 `--train-stage all` 会训练，请在验证时显式选择轻量阶段。

训练参数可用 `GSV_S2_BATCH`、`GSV_S2_EPOCHS`、`GSV_S1_BATCH`、`GSV_S1_EPOCHS` 调整。脚本默认参数来自有限显存案例，不是质量保证或显存通用上限。权重保存在上游 `SoVITS_weights_v2` / `GPT_weights_v2`；临时配置位于 `TEMP`，按实验名区分。

## 已有检查点配音

```powershell
& $env:GSV_PYTHON "$B/infer_tts.py" --root $env:GSV_ROOT --sovits 'E:/Models/character.pth' --gpt 'E:/Models/character.ckpt' --ref "$W/ref/example.wav" --prompt-text '参考音频的准确原文。' --text '需要新生成的句子。' --prompt-lang ja --lang ja --out "$W/out/tts_001.wav"
```

参考 3–10 秒；输出 WAV 保留合成器原始起音与尾音，JSON 记录哈希和版本，不覆盖已有结果。`--device cpu` 可避开其它 GPU 任务，速度较慢。上游推理库会改写 `weight.json`，驱动在正常退出或 Python 异常时恢复原字节；外部强制终止无法保证 finally 执行。不要与另一个 GPT-SoVITS 模型选择会话同时运行。不要通过选择最高相似度或单个 seed 宣称模型质量达标。实际验收需检查完整朗读、尾字、语种、音色及伪影。

## DSH 验证方式

在新会话自然语言说明：加载两个技能、读取本地案例说明、只导出少量已有片段、用显式实验和训练集执行 check/list、输出报告、在 GPU 空闲时用已有检查点生成一句新台词。保留宿主的技能读取、工具命令、退出码、产物哈希和状态记录。不要把独立 CLI 的成功记成 DSH 成功。

GPU 忙时先做 CPU 验证，不停止其它任务。不为冒烟测试重新训练全量模型。详细已测项、失败项、未测项见 `validation.md`。

## 本轮宿主环境注意事项

此次 DSH 宿主先遇到 `grantWrite` 的 Windows ACL 授权错误，由用户在 DSH 中处理后继续。本轮成功步骤运行于 `danger-full-access` 模式，`workspace-write` 模式的恢复效果仍需单独验证。技能自身不应自动修改目录 ACL 或宿主安全设置。

此次 DSH 继承的 `NO_PROXY` IPv6 项触发上游 Gradio/httpx 的 `Invalid port: ':1]'`，影响发生在推理模块导入时。已验证的临时办法是在运行该本地推理子进程时传入 `NO_PROXY=localhost,127.0.0.1`。例如在 Python 编排器中：

```python
child_env = dict(os.environ, NO_PROXY="localhost,127.0.0.1")
subprocess.run(command_argv, env=child_env, check=True)
```

`command_argv` 为原推理参数列表。此办法仅用于已经观察到该错误的本地验证；保留原始错误和环境覆盖记录，勿把临时覆盖写成全局代理配置。完整带参数的脱敏命令见验证记录中的宿主回执。
