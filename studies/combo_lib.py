"""QQQ 腿的 RV 降风险回测：tearsheet 与 combo_comparison 共用，避免各写一份。

这是**回测侧**的向量化实现；实盘侧是 live/leg_signal.py（标量实现）。
两者的逐位一致性由 tools/leg_equivalence.py 验证（全历史每隔 7 天抽样 + 最近 120 天，0.00e+00）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import round3_common as C
from sens_rv_weights import pct_w, w_linear, w_thresh

DEFAULT_LO, DEFAULT_HI = 0.5, 1.5          # 选定的"浅蓝圆"
COST_BP = 2.0


def rv_weight_series(index: pd.DatetimeIndex, lo=DEFAULT_LO, hi=DEFAULT_HI, kind="linear",
                     rvw=21, pw=504) -> pd.Series:
    """返回 index 上每天**实际作用**的 QQQ 腿权重（t−1 收盘的信号 -> t 日）。

    信号在 QQQ 全历史（1999 起）上算，所以 2019 年起点已有充足预热。
    """
    r = C.load_close("yahoo/QQQ.parquet").pct_change()
    p = pct_w(r.rolling(rvw).std(), pw)
    w = w_linear(p, lo, hi) if kind == "linear" else w_thresh(p, lo, hi)
    return w.shift(1).reindex(index).fillna(1.0)


def qqq_leg(bh: pd.Series, q: float = 1.0, lo=DEFAULT_LO, hi=DEFAULT_HI, kind="linear",
            cost_bp: float = COST_BP) -> tuple[pd.Series, pd.Series]:
    """QQQ 腿收益 = q·w·r − 成本。返回 (腿收益, 权重序列)。"""
    w = rv_weight_series(bh.index, lo, hi, kind)
    turn = w.diff().abs().fillna(0.0)
    return q * w * bh - cost_bp / 1e4 * q * turn, w
