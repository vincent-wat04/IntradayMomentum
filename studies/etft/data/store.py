"""最小化 vendor：intraday 引擎只需要 raw_yahoo（读日频复权价算除息调整的前收）。
数据位置：$ETFT_DATA/raw/yahoo/{ticker}.parquet"""
import functools

import pandas as pd

from etft import config as C


def _yfile(t: str) -> str:
    return str(C.RAW / 'yahoo' / f"{t.replace('^', 'IDX_').replace('=', '_')}.parquet")


@functools.lru_cache(maxsize=None)
def raw_yahoo(t: str) -> pd.DataFrame:
    df = pd.read_parquet(_yfile(t))
    df = df[~df.index.duplicated(keep='last')].sort_index()
    for c in ('open', 'high', 'low'):
        bad = (df[c] <= 0) | df[c].isna()
        df.loc[bad, c] = df.loc[bad, 'close']
    return df
