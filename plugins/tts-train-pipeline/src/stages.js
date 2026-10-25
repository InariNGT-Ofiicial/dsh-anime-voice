/**
 * 流程与决策点。文本集中在这里，index.js 只负责搬运。
 *
 * ★ 核心立场：**能用原文就绝不用 ASR**。
 *   语料本身带台词（剧本、字幕、标注表）时，那是引擎级的无损数据；
 *   ASR 是"重新听写一遍"，会丢标点、认错人名、写错汉字。
 *   实测同一部 galgame：剧本提取 3135 条精确对齐，之前按菜单解析只有 102 条。
 *   所以 ASR 只用于「真的没有任何文本」的语料。
 */

export const FLOW = [
  {
    id: '1. 找回原文',
    tool: 'tts_script',
    what: '从剧本反编译产物里抽「说话人 + 台词 + 语音文件名」。KrKrZ 的台词在 scene.texts 里，'
      + '不在 lines 里——这正是按 lines 找文本只能捞到菜单的原因。',
    fail: '只有 .scn 二进制而没有反编译出的 .json → 先解包（dsh-train-pipeline 的 train_unpack）',
    first: true,
  },
  {
    id: '2. 盘语料',
    tool: 'tts_scan',
    what: '分清「多少语音文件」与「多少条可训数据」。台词与语音都要有才算一条。',
    fail: '语音命中率低说明 voice ID 与文件命名对不上，先看未命中样本再决定',
  },
  {
    id: '3. 兜底补台词',
    tool: 'tts_asr',
    what: '**只在真的没有文本时用**：faster-whisper 逐条转写，按置信度过滤。',
    fail: '日文 ASR 准确率有限，低置信度句子会污染音色；能找回原文就别走这一步',
    fallback: true,
  },
  {
    id: '4. 出料',
    tool: 'tts_build',
    what: '32kHz 单声道 PCM16 + list.txt（wav名|说话人|语言|文本）。优先用精确 voice_path。',
    fail: '★ 超长不切分：切了音频和文本就对不上，只能整条丢弃',
  },
  {
    id: '5. 预处理',
    tool: 'tts_preprocess',
    what: '1-get-text（音素 + BERT）→ 2-get-hubert（SSL + 32k）→ '
      + '【v2Pro/v2ProPlus 专属】2-get-sv（7-sv_cn 说话人向量）→ 3-get-semantic（语义 token）',
    fail: '脚本只读环境变量；有残留产物会直接跳过，换数据集必须先重置。'
      + 'Pro 版本漏了 2-get-sv 不会报错，但 s2 会把 spec/ssl/wav 一起换成全零去训',
  },
  {
    id: '6. 训练',
    tool: 'tts_run',
    what: 's2（SoVITS 声学）出 .pth → s1（GPT 语义）出 .ckpt。顺序不可倒。',
    fail: '显存不足先降 batch_size；epoch 数小于保存间隔会一个权重都不落',
  },
  {
    id: '7. 试听与注册',
    tool: 'tts_infer → tts_register',
    what: '合成一句 →（可选）ASR 回读核对 → 写进音色清单',
    fail: '合成出来是静音/杂音，多半是参考音频太短或与目标音色差太远',
  },
];

export const DECISIONS = [
  {
    id: 'D1 · 文本从哪来（**最重要的一步**）',
    options: [
      '① 剧本反编译 JSON（.ks.json / .scn.json）→ tts_script，**首选**：说话人、原文、语音 ID 全有',
      '② 旁挂字幕 / 标注表（.jsonl / .tsv / .srt）→ 直接喂 tts_build 的 lines',
      '③ 语音目录旁有同名 .txt → 名字对齐即可',
      '④ 以上都没有 → 才用 tts_asr 兜底',
    ],
    rule: '★ 顺序不能反。用 ASR 去替换已有的原文，等于拿有损数据换无损数据：'
      + '日文里「はし」和「橋」听感相同，ASR 会写错汉字，模型就学错发音。',
  },
  {
    id: 'D2 · 文本语言必须与语音语言一致',
    options: [
      '日文语音 → 只能用日文原文（KrKrZ 的 texts[.][2][0]）',
      '中文语音 → 用中文文本',
      '剧本里常有官方中文翻译（texts[.][2][2]）——那是**给人看的**，不能拿去训日文语音',
    ],
    rule: '★ 中日混训会让模型学出「中文字形配日文音」的怪物。翻译列只在语音本身是中文时才有用。',
  },
  {
    id: 'D3 · 版本选哪个',
    options: [
      'v2ProPlus（默认）：本机已装底模，推理显存最低（实测 1.75 GB）',
      'v2Pro：更省一点，音质略差',
      'v3/v4：音质最好但训练脚本换成 s2_train_v3_lora.py，且没有 s2D 底模',
    ],
    rule: 'v2Pro/v2ProPlus 的 s1 底模是 s1v3.ckpt，v2 是 v2final 的 5kh；用错能训但收敛差。',
  },
  {
    id: 'D4 · 语料时长够不够',
    options: [
      '<5 min → 不建议训，音色会飘',
      '5~20 min → 能出人声，辨识度有限',
      '20~60 min → 可用（甜点区）',
      '>60 min → 充足，训练时间线性增长',
    ],
    rule: '时长按**可训条数**算，不是目录里有多少文件。',
  },
  {
    id: 'D5 · 显存不足怎么办',
    options: [
      '先降 s2BatchSize（4 → 2 → 1）',
      '开 gradCkpt=true（省显存，慢一些）',
      '把 s2Epochs 降下来先验证链路，成功后再拉长',
    ],
    rule: '不要靠关 fp16 省显存：s2 在 fp32 下显存反而更高。',
  },
];

export const RULES = [
  '★ **原文优先，ASR 兜底**。有文本却去跑 ASR 是返工，不是保险。',
  '★ 文本语言必须与语音语言一致。剧本里的中文翻译不能拿去训日文语音。',
  '★ 音频与文本必须逐条对齐。任何切分/拼接都会让监督信号错位，比少几条数据更糟。',
  '★ list.txt 每行必须正好 4 段（| 分隔）；文本里的 | 会被替换成全角逗号。',
  '★ 预处理产物带「已存在就跳过」逻辑：换数据集必须先重置 opt_dir。',
  '★ v2Pro/v2ProPlus 必须有 7-sv_cn（2-get-sv 的产物）。缺了上游 data_utils 会'
    + '把 spec/ssl/wav 一起换成全零继续训，不报错但训出来是废的 —— 这是 Silent Fail 里的头号陷阱。',
  '★ 日文语料的 3-bert 是空目录，正常：1-get-text 只在 lan=="zh" 时才抽 BERT。',
  '★ 训练进程的 cwd 必须是工程根：权重输出目录是相对路径，换个 cwd 会写到别处且不报错。',
  '★ 后台训练必须 PYTHONUNBUFFERED=1：Python 对非 tty 的 stdout 是块缓冲，'
    + '不设则日志长时间 0 字节，崩溃时缓冲区还不刷，真正的报错直接丢。',
  '★ 端到端验证要 ASR 回读：能出声 ≠ 念对了。',
  '★ 欠拟合时先调采样再谈加训：temperature 1.0→0.6、top_k 15→5 能救掉大部分「长句中间跳字」，'
    + '整段朗读再用 split=cut0，别一上来就重训。',
  '★ s1 只训几轮是纯负收益：它把底模的多语言能力削掉了，母语也没练精。'
    + '要么训够，要么不训 —— 跨语言（日文语料念中文）必须用底模 s1：tts_infer gptMode=base。'
    + '音色来自 s2，发音能力来自 s1，这两件事是分开的。',
];

export const VERSION_NOTES = {
  v1: '老版本，s1 底模是 2kh 的，除非复现旧结果否则不用',
  v2: '稳定版，s1 用 v2final 5kh',
  v2Pro: '推荐；s1 底模是 s1v3.ckpt（不是 v2final）',
  v2ProPlus: '默认推荐；本机已装底模，推理显存最低；s1 底模是 s1v3.ckpt',
  v3: '音质更好，训练脚本不同（s2_train_v3_lora.py），无 s2D',
  v4: '同上，底模 s2Gv4.pth，需另外的 vocoder',
};

/** 剧本引擎的已知结构说明（写进 tts_script 的输出，让人知道它凭什么能抽到） */
export const SCN_STRUCTURE = {
  engine: 'KrKrZ（柚子社）',
  where_text_lives: 'scene.texts[i][2][<lang>][1]',
  where_speaker: 'scene.texts[i][0]',
  where_voice: 'scene.texts[i][3][0].voice',
  why_lines_fails: 'lines 里只有指令流（bgm/立绘/env），台词在场景级 texts 里通过索引引用；'
    + '在 lines 里找文本只能捞到选项菜单',
  lang_columns: '0=日文原文, 1=英文, 2=简体中文, 3=繁体中文',
};
