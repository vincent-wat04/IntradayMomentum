"""Vendored 配置。数据目录可用环境变量覆盖，默认指向本库的 data/。"""
import os
from pathlib import Path

ROOT = Path(os.environ.get('ETFT_HOME', Path(__file__).resolve().parents[2].parent))
DATA = Path(os.environ.get('ETFT_DATA', ROOT / 'data'))
RAW = DATA / 'raw'
PROC = DATA / 'processed'
RESULTS = ROOT / 'results'
