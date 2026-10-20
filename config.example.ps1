# Copy to config.local.ps1; edit paths, then dot-source it. No automatic install/download.
$env:ANIME_PYTHON = 'E:/Voice/GPT-SoVITS/.venv/Scripts/python.exe'
$env:GSV_ROOT = 'E:/Voice/GPT-SoVITS'
$env:GSV_PYTHON = $env:ANIME_PYTHON
$env:FFMPEG_BIN = 'E:/Tools/ffmpeg/bin/ffmpeg.exe'
$env:HF_HOME = 'E:/Models/huggingface'
$env:MODELSCOPE_CACHE = 'E:/Models/modelscope'
$env:ANIME_WHISPER_MODEL = 'E:/Models/faster-whisper-large-v3'
$env:ANIME_CAMPP_MODEL = 'E:/Models/campplus/snapshots'
$env:ANIME_SEP_ROOT = 'E:/Voice/MSST'
$env:ANIME_SEP_PYTHON = 'E:/Voice/MSST/env/python.exe'
$env:ANIME_SEP_CONFIG = 'configs/config_speech_bs_roformer_AliceN.yaml'
$env:ANIME_SEP_CHECKPOINT = 'pretrain/speech_bs_roformer_AliceN_sdr_20.4440.ckpt'
# Never default to a prior project. Set these explicitly for each training run.
# $env:GSV_EXP = 'my_character_v1'
# $env:GSV_TRAINSET = 'E:/Projects/my_character/_full/out/ts'
