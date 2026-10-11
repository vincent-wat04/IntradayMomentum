"""审计工具箱：用**机械化测试**抓整类错误，而不是靠读代码。

读代码找 bug 已经被证明不可靠（本项目里我自己漏掉过三次：逐 bar 累加的前视、
bar 时标偏移、杠杆 ETF 的名义金额）。这个模块提供的是能自动抓住整类错误的手段：

1. poison_future()      未来数据投毒 —— 抓任何形式的前视
2. poison_intraday()    日内未来 bar 投毒 —— 抓 bar 级前视（PART8 那个 bug）
3. 不变量/等变量检查     抓记账错误（成本、杠杆、符号对称性）
4. 安慰剂               抓"看起来有效其实是机械产物"

设计原则：每个测试都要**在已知的 bug 上失败**。所以每个函数下面都注明它能抓住
本项目真实发生过的哪个错误。如果一个测试抓不住任何已知错误，它就是摆设。
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 投毒

def poison_future(b, cut_day, factor=7.3):
    """把 cut_day **之后**的所有价量数据乘以一个夸张的倍数。

    任何使用了未来数据的计算，在 cut_day 及之前的输出都会因此改变。
    能抓住：全样本标准化、居中窗口、bfill、滚动窗口忘了 shift(1)、
            标签泄漏、用未来的除权因子。
    抓不住：同一根 bar 内的前视（用 poison_intraday）。
    """
    out = copy.copy(b)
    m = b.c.index > pd.Timestamp(cut_day)
    for k in ("o", "h", "l", "c"):
        X = getattr(b, k).copy()
        X.loc[m] = X.loc[m] * factor
        setattr(out, k, X)
    V = b.v.copy(); V.loc[m] = V.loc[m] * factor; out.v = V
    for k in ("day_open", "day_close", "prev_close_adj"):
        s = getattr(b, k).copy()
        mm = s.index > pd.Timestamp(cut_day)
        s.loc[mm] = s.loc[mm] * factor
        setattr(out, k, s)
    return out


def poison_intraday(b, cut_clock, end_clock="15:30", factor=7.3):
    """把**每一天** (cut_clock, end_clock] 区间内的 bar 乘以夸张倍数。

    刻意**不碰当日最后几根 bar**，这样 day_close / prev_close_adj / 日波动率
    都不变 —— 否则仓位大小会因为日级数据被污染而合法地改变，测试就失效了。

    决策在第 k 根 bar 收盘、成交在 op[k+1]，所以污染 cut_clock 之后不应改变
    cut_clock 之前的**决策**（方向/时点/仓位）。P&L 允许变（出场在后面）。
    能抓住：PART8 那个 `pnl += pos*(op[j+1]/op[j]-1)` 的逐 bar 累加前视、
            止损用了未来的 high/low、用了全天 VWAP 而不是累计 VWAP。
    """
    out = copy.copy(b)
    cols = [c for c in b.c.columns if cut_clock < c <= end_clock]
    for k in ("o", "h", "l", "c"):
        X = getattr(b, k).copy()
        X[cols] = X[cols] * factor
        setattr(out, k, X)
    V = b.v.copy(); V[cols] = V[cols] * factor; out.v = V
    return out


def shift_prices(b, mult):
    """所有价格乘以常数（股数/每股成本会变，收益率不变）。
    能抓住：把"每股固定成本"当成"按名义金额比例的成本"，或者反过来。
            —— 这正是 PART8 的 TQQQ 错误那一类。"""
    out = copy.copy(b)
    for k in ("o", "h", "l", "c"):
        setattr(out, k, getattr(b, k) * mult)
    for k in ("day_open", "day_close", "prev_close_adj"):
        setattr(out, k, getattr(b, k) * mult)
    return out


# ---------------------------------------------------------------- 比较

def cmp_frames(a, b, cols, tol=1e-10, label=""):
    """返回 (是否相同, 说明)。"""
    idx = a.index.intersection(b.index)
    bad = {}
    for c in cols:
        if c not in a.columns or c not in b.columns:
            continue
        x, y = a.loc[idx, c], b.loc[idx, c]
        both = x.notna() & y.notna()
        d = (x[both] - y[both]).abs()
        den = 1e-12 + y[both].abs()
        rel = (d / den).max() if both.any() else 0.0
        nanmis = int((x.isna() != y.isna()).sum())
        if (rel > tol) or nanmis:
            bad[c] = dict(max_rel_diff=float(rel), nan_mismatch=nanmis,
                          n_diff=int((d > tol * den).sum()))
    return (not bad), bad


def cmp_trades(ta, tb, upto_bar, keys=("day", "bar", "dir", "size")):
    """只比较 bar <= upto_bar 的交易记录（决策层面）。"""
    if ta is None or tb is None or len(ta) == 0 or len(tb) == 0:
        return False, "no trade records"
    A = ta[ta.bar <= upto_bar].reset_index(drop=True)
    B = tb[tb.bar <= upto_bar].reset_index(drop=True)
    if len(A) != len(B):
        return False, f"交易笔数不同: {len(A)} vs {len(B)}"
    for k in keys:
        if k not in A.columns:
            continue
        if A[k].dtype.kind in "fc":
            if not np.allclose(A[k].to_numpy(), B[k].to_numpy(), rtol=1e-10, atol=1e-12):
                return False, f"字段 {k} 不同"
        elif not A[k].equals(B[k]):
            return False, f"字段 {k} 不同"
    return True, f"{len(A)} 笔决策完全一致"


# ---------------------------------------------------------------- 绩效

def perf(x):
    x = pd.Series(x).dropna()
    if len(x) < 30 or x.std() == 0:
        return dict(sharpe=np.nan, ann=np.nan, n=len(x))
    return dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)),
                ann=float(x.mean() * 252), vol=float(x.std() * np.sqrt(252)),
                n=int(len(x)))
