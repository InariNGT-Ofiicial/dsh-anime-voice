# DSH Anime Voice

番剧台词提取与 GPT-SoVITS v2 训练/配音的两个本地技能。与 DDSP 训练、歌曲翻唱项目分别维护。

2026-10-20 已核验一条真实 DSH 代表性链路：自然语言任务 → 技能加载 → 少量真实片段导出 → 数据集校验/清单 → 报告 → 已有检查点 CPU 配音。运行时使用 `danger-full-access` 授权，并调整了推理子进程的 `NO_PROXY`；受限模式、完整训练和人工听音仍待验证。见 [宿主回执与验收范围](docs/validation.md#真实-dsh-回执复核)。

| 技能 | 输入 | 输出 |
|---|---|---|
| `anime-voice-line-extractor` | 本地音视频、已确认的角色映射 | 角色 WAV/TXT、绝对时间轴 CSV、`train.list` |
| `anime-voice-clone-training` | 整集音频、人工确认的声纹参照、明确模型路径 | 整集重建数据集、训练特征/权重、单句配音 WAV/JSON |

A 可以直接导出 `绝对wav路径|角色|语言|文本`。B 的主路线复用 A 建声纹参照，再从整集分离和带标点 ASR 重建数据集；不是把 A 的旧 157 条直接洗成新 291 条。训练器也可以独立读取 A 的清单，但素材是否适合训练仍需审核。

## 安装

```powershell
git clone https://github.com/InariNGT-Ofiicial/dsh-anime-voice.git
cd dsh-anime-voice
./install.ps1 -SkillRoot "$env:USERPROFILE/.dsh/skills"
```

已经安装时加 `-Update`；脚本会先备份这两个技能。若 `.dsh/skills` 是 Junction，会写入它实际指向的技能根。安装不修改宿主配置、模型、Python 环境或其它技能。

Windows 11、PowerShell 7 和 Python 3.10–3.12 是本项目的目标环境。分析侧依赖见 `requirements-analysis.txt`；训练和推理必须使用已经配置好的 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v2 环境。分离使用独立的 [MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training) 兼容环境。不要把这几套环境混装。

复制 `config.example.ps1` 为本地配置，填入自己的路径后执行：

```powershell
. ./config.local.ps1
$A = "$env:USERPROFILE/.dsh/skills/anime-voice-line-extractor/scripts"
$B = "$env:USERPROFILE/.dsh/skills/anime-voice-clone-training/scripts"
& $env:ANIME_PYTHON "$A/anime_lines.py" --help
& $env:ANIME_PYTHON "$B/run_pipeline.py" --list-stages
```

完整操作、依赖和可复现命令见 [运行说明](docs/usage.md)，宿主实测范围见 [验证记录](docs/validation.md)。在 DSH 新会话里用自然语言说明素材路径、角色、输出目录，以及允许哪些耗时阶段。技能目录存在只证明文件已安装；真实宿主验收必须包含模型选择技能、调用工具和生成产物的证据。

## 能力边界

声纹、F0、ASR 一致性都是辅助筛选，不能替代人工听音与文本校对；默认 F0 下限为 0，必须按目标声音调整。标点只能说明文字形态，不能证明音频自然收尾。脚本生成 WAV 不等于朗读正确，跨语种尤其需要逐句复听。

没有通用的“25/38 分钟就达标”门槛，也不把余弦相似度解释成“不可区分”。旧版本的声纹/F0数字不代表当前权重已通过验收。当前维护验证优先复用已存在的 v2 检查点，不重新训练整个模型或重跑三集全量流程。

仓库只包含代码和脱敏技术验证记录；不包含番剧、声音素材、模型权重、用户会话或凭据。素材、声音及模型的使用授权由使用者确认，各上游依赖分别遵循其许可证。代码尚未指定开放许可证。

## 维护

```powershell
python -m unittest discover -s tests -v
```

问题反馈请提供脚本版本、已脱敏的命令/退出码、阶段、期望和实际产物。不要上传模型、私有路径、音频或账号配置。CPU 接口回归由 GitHub Actions 执行；GPU 和 DSH 宿主验证记录单独维护，不能由 CI 通过替代。
