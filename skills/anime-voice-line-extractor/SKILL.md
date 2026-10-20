---
name: anime-voice-line-extractor
description: 从本地番剧或动画音频按说话人提取台词、角色 WAV/TXT 和 GPT-SoVITS 清单；用于台词本、角色切片和训练参照准备。整集重建训练集与训练配音使用 anime-voice-clone-training。
---

# 番剧角色台词提取

先确认输入、语言、目标角色、输出目录和需排除的 OP/ED 区间。脚本位于本技能的 `scripts/`，用已配置的分析 Python；`FFMPEG_BIN`、`ANIME_WHISPER_MODEL`、`ANIME_CAMPP_MODEL` 配置本地依赖。不重装用户的现有环境。

1. `profile_episode.py INPUT --block 30` 与 `stereo_probe.py INPUT --start 80 --end 200 --step 2` 辅助定位片段；边界由人工确认，不假定其它作品的片头长度。
2. `anime_lines.py run INPUT -o RUN --language ja --keep "0:00-1:30,2:40-22:00"` 输出角色簇、CSV、词级时间戳与报告。时间戳保持源片绝对时间。
3. 听样本并创建映射，再运行 `anime_lines.py rename RUN --map speaker_map.json`；同人多簇用 `refine RUN --map roles.json --target ROLE -o REFINED`。跨集簇号不代表同一人，可用 `anchor_speakers.py` 提供候选匹配，再按文本和声音确认。
4. `export_trainset.py REFINED --src RUN --out TRAINSET --min-dur 1.5` 导出 `train.list` 与待校对 TSV。多集分别导出，用 `merge_trainset.py PARENT --order ep01,ep02` 合并。
5. 试听 `gen_preview.py REFINED` 生成的页面，并逐条核对角色、转写和切片边界。

清单格式是 UTF-8，每行 `绝对wav路径|角色|语言|文本`。角色和文本不应包含分隔符 `|`。保留输入、映射、配置、命令与退出码；工作产物放项目目录，不能写进技能目录。

`dry_vocals.py`、`make_sep_packs.py` 与损伤分析脚本保留供局部处理/诊断；不要据此声称已重建整集训练集。训练主路线由 B 使用整集干声和带标点 ASR 重建。ASR 回归可能误判喊叫；声纹不能可靠区分相近声线。阈值只是候选筛选，必须人工复听。

涉及 GPU 时先检查现有任务，不能自动停止别的任务。技术导出成功不等于训练质量通过；不得把旧案例时长、声纹或 F0 结果当成新权重的验收结论。
