"""数据与引擎路径的唯一出口。所有脚本从这里取路径，不再写死本机目录。

ETFT_DATA   原始数据根目录（内含 raw/min1、raw/cboe、raw/yahoo ...），默认 <库根>/data
ETFT_HOME   etf_timing 全量工程位置（仅少数脚本需要读它的 results/），默认库根
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ETFT = Path(os.environ.get('ETFT_HOME', ROOT))
DATA = Path(os.environ.get('ETFT_DATA', ROOT / 'data'))
RAW = DATA / 'raw'
