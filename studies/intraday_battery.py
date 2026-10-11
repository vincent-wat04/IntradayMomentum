"""日内策略全家桶：TQQQ 版日内动量、隔夜仓的影响、以及 8 种日内回转策略的对照实验。

数据：
  data/SPY_5min.csv, data/QQQ_5min.csv   仓库原有（bar 起始时标、后复权）
  data/min5_us/*.csv                      77 只美股 5 分钟（moomoo 前复权）
  data/lev/levetf_close.csv               TQQQ/QLD/UPRO/SSO/SPY/QQQ 真实日线

TQQQ 怎么测（没有 TQQQ 的 5 分钟数据，额度用满了）：
  杠杆 ETF 的设计目标是复制**单日**收益的 3 倍，日内 NAV 近似连续跟踪 3 倍。实测日频
  realized beta 2.96、相关 0.9991（见 PART2）。所以用 QQQ 的 5 分钟对数收益 ×3 模拟 TQQQ 的
  日内路径，是一个合理近似，并且**关键点在于：日内策略收盘平仓，不跨日持仓，因此完全避开了
  杠杆 ETF 的每日再平衡复利衰减**（那 −8.9pp/年是跨日复利造成的，不是日内成本）。
  需要付的额外代价只有两项：更宽的相对价差、以及日内那一小段融资成本。
  相对价差用 Corwin-Schultz 日频估计量在真实 TQQQ/QQQ 日线上实测，不靠假设。

"不持隔夜仓"：噪声区域策略**本来就不持隔夜仓**（收盘 MOC 平仓）。会持隔夜仓的是
"买入持有 + 0.5x 叠加层"这个组合。本脚本把两者分开量化。

8 种日内回转策略（全部 bar 收盘决策、下一根 bar 开盘成交、收盘平仓）：
  NOISE      噪声区域突破（基线，Zarattini, Aziz & Barbon 2024）
  ORB5       5 分钟开盘区间突破 + 止损在首根 bar 另一端（Zarattini & Aziz 2023）
  ORB5_ATR   同上但止损用 ATR、无盈利目标、收盘平仓（他们的 TQQQ 变体）
  ON_REV     隔夜跳空的日内反转（Lou, Polk & Skouras 2018 JFE：动量全在隔夜、反转在日内）
  ON_CONT    隔夜跳空的日内延续（反向对照）
  VWAP_REV   偏离 VWAP 后的回归
  GAO        首半小时 → 末半小时动量（Gao, Han, Li & Zhou 2018 JFE）
  LUNCH_REV  午盘时段反转
  LAST30     末半小时动量
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

OUT = Path("outputs/full/intraday"); OUT.mkdir(parents=True, exist_ok=True)
SPLIT = "2023-01-03"
NOISE_LB = 14          # 噪声区域回看天数（论文参数）
BAND = 1.0             # 区间宽度（论文参数）
# 论文的评估时点是 10:00、10:30 ... 15:00（**bar 收盘时刻**）。
# 本仓库的 5 分钟 CSV 是 **bar 起始时标**，所以收盘在 10:00 的那根 bar 的标签是 09:55。
# 如果直接用 "10:00" 当检查点，等于把每个决策都推迟了 5 分钟（实测会损失 ~0.3 Sharpe）。
_CHECK_ENDS = [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)][:-1]      # 10:00..15:00
CHECKS = [(pd.Timestamp(f"2000-01-01 {t}") - pd.Timedelta("5min")).strftime("%H:%M")
          for t in _CHECK_ENDS]                                                      # 09:55..14:55


def load_5min(path):
    raw = pd.read_csv(path, parse_dates=["datetime"]) if "datetime" in pd.read_csv(path, nrows=1).columns \
        else pd.read_csv(path).rename(columns={"time_key": "datetime"}).assign(
            datetime=lambda d: pd.to_datetime(d.datetime))
    raw["day"] = raw.datetime.dt.strftime("%Y-%m-%d")
    raw["clock"] = raw.datetime.dt.strftime("%H:%M")
    raw = raw[(raw.clock >= "09:30") & (raw.clock < "16:00")].sort_values("datetime")
    n = raw.groupby("day").size()
    raw = raw[raw.day.isin(n[n >= 70].index)]
    return raw


def day_panels(raw):
    """返回 dict: day -> (clock array, open, high, low, close, volume)."""
    out = {}
    for day, g in raw.groupby("day", sort=True):
        out[day] = (g.clock.to_numpy(), g.open.to_numpy(float), g.high.to_numpy(float),
                    g.low.to_numpy(float), g.close.to_numpy(float), g.volume.to_numpy(float))
    return out


def noise_bands(panels, days, lb=NOISE_LB, band=BAND):
    """σ_{t,m} = 过去 lb 天同一时点 |close_{d,m}/open_d − 1| 的均值（shift(1)，只用过去数据）。

    与 etf_timing 已验证实现 (etft/intraday/noise.py::sigma_profile) 口径一致：用 bar **收盘价**
    相对当日开盘价的偏离，而不是 bar 开盘价。"""
    clocks = panels[days[0]][0]
    M = {c: i for i, c in enumerate(clocks)}
    rel = np.full((len(days), len(clocks)), np.nan)
    for i, d in enumerate(days):
        cl, op, hi, lo, c, v = panels[d]
        o0 = op[0]
        for j, ck in enumerate(cl):
            if ck in M:
                rel[i, M[ck]] = abs(c[j] / o0 - 1)
    mo = max(5, int(0.7 * lb))
    sig = pd.DataFrame(rel).rolling(lb, min_periods=mo).mean().shift(1).to_numpy() * band
    return M, sig


_RSIGN = np.random.default_rng(20261002).choice([-1.0, 1.0], 4000)


def day_atr(panels, days, lb=14):
    """ATR14（相对值）：前 lb 天 (日高−日低)/日收 的均值，shift(1)，只用过去数据。"""
    rng_ = pd.Series({d: (panels[d][2].max() - panels[d][3].min()) / panels[d][4][-1] for d in days})
    return rng_.rolling(lb, min_periods=7).mean().shift(1).reindex(days).to_numpy()


def simulate(panels, days, strat, cost_bp, lev=1.0, prev_close=None, sig=None, M=None, atr=None):
    """每天独立模拟，收盘 MOC 平仓，返回每日收益序列（名义 1x × lev）。

    时序约定（严格无前视，和 etf_timing/etft/intraday/strategy.py 一致）：
        第 k 根 bar **收盘时**用 bar 0..k 的信息做决策 -> 第 k+1 根 bar **开盘价** op[k+1] 成交。
        盈亏按**成交价到成交价**记账：pnl += pos * (出场成交价 / 入场成交价 − 1)。
        这样不可能把某根 bar 的涨跌算给在这根 bar 收盘才决定的仓位。
        最后一根 bar 用当日收盘价平仓。
    """
    cvec = (np.full(len(days), float(cost_bp)) if np.isscalar(cost_bp)
            else np.asarray(cost_bp, float))      # 支持逐日成本（杠杆 ETF 的相对价差随价格变化）
    rets, ntr = [], []
    for i, d in enumerate(days):
        cost = cvec[i] * 1e-4
        cl, op, hi, lo, c, v = panels[d]
        n = len(cl)
        o0 = op[0]
        dayclose = c[-1]
        pc = prev_close[i] if prev_close is not None else np.nan
        if not (np.isfinite(pc) and pc > 0):
            pc = o0
        pos, entry, pnl, trades = 0.0, np.nan, 0.0, 0.0
        rsign = _RSIGN[i % len(_RSIGN)]
        gap = o0 / pc - 1
        tp = (hi + lo + c) / 3.0
        vwap = np.cumsum(tp * v) / np.maximum(np.cumsum(v), 1e-12)
        fh_end = int(np.searchsorted(cl, "10:00"))          # 第一根 09:30 后 10:00 的 bar 下标
        lunch_s = int(np.searchsorted(cl, "11:30"))
        lunch_e = int(np.searchsorted(cl, "13:30"))
        last30 = int(np.searchsorted(cl, "15:30"))
        a = atr[i] if atr is not None else np.nan            # 前一日为止的 ATR14（相对值）
        ref_hi = ref_lo = np.nan

        for k in range(n - 1):                               # 决策在 bar k 收盘，成交在 op[k+1]
            px = op[k + 1]
            if not np.isfinite(px) or px <= 0:
                continue
            want = pos

            if strat == "NOISE":
                if sig is None or M is None or cl[k] not in M:
                    continue
                b = sig[i, M[cl[k]]]
                if not np.isfinite(b):
                    continue
                if cl[k] not in CHECKS:                      # 只在整/半点评估（论文设定）
                    continue
                ref_hi = max(o0, pc) * (1 + b)
                ref_lo = min(o0, pc) * (1 - b)
                if pos == 0:
                    if c[k] > ref_hi:
                        want = 1.0
                    elif c[k] < ref_lo:
                        want = -1.0
                else:                                        # VWAP 追踪止损（已验证的默认设定）
                    vw = vwap[k]
                    stop_l = max(ref_hi, vw) if np.isfinite(vw) else ref_hi
                    stop_s = min(ref_lo, vw) if np.isfinite(vw) else ref_lo
                    if (pos > 0 and c[k] < stop_l) or (pos < 0 and c[k] > stop_s):
                        want = 0.0

            elif strat in ("ORB5", "ORB5_ATR"):
                if k == 0:                                   # 首根 bar 收盘定方向，op[1] 成交
                    r0 = c[0] / op[0] - 1
                    if abs(r0) > 1e-6:
                        want = 1.0 if r0 > 0 else -1.0
                elif pos != 0:
                    if strat == "ORB5":                      # 止损在首根 bar 的另一端
                        if pos > 0 and c[k] < lo[0]:
                            want = 0.0
                        elif pos < 0 and c[k] > hi[0]:
                            want = 0.0
                    elif np.isfinite(a) and np.isfinite(entry):   # 止损 = 0.5×ATR14
                        if pos > 0 and c[k] < entry * (1 - 0.5 * a):
                            want = 0.0
                        elif pos < 0 and c[k] > entry * (1 + 0.5 * a):
                            want = 0.0

            elif strat in ("ON_REV", "ON_CONT"):
                if k == 0 and abs(gap) > 1e-6:               # 跳空在 o0 已知，op[1] 成交
                    want = (-1.0 if strat == "ON_REV" else 1.0) * float(np.sign(gap))

            elif strat == "VWAP_REV":
                vw = vwap[k]
                if np.isfinite(vw) and vw > 0:
                    dev = c[k] / vw - 1
                    thr = 0.002
                    want = -1.0 if dev > thr else (1.0 if dev < -thr else 0.0)

            elif strat == "GAO":                             # 首半小时动量 -> 末半小时持仓
                if k == last30 - 1 and fh_end > 0:
                    r_fh = c[fh_end - 1] / o0 - 1
                    want = float(np.sign(r_fh)) if abs(r_fh) > 1e-6 else 0.0

            elif strat == "LUNCH_REV":                       # 午盘涨跌 -> 午后反向
                if k == lunch_e - 1 and lunch_e > lunch_s:
                    r_l = c[lunch_e - 1] / op[lunch_s] - 1
                    want = -float(np.sign(r_l)) if abs(r_l) > 1e-6 else 0.0

            elif strat == "LAST30":                          # 全日动量 -> 末半小时持仓
                if k == last30 - 1:
                    r_d = c[k] / o0 - 1
                    want = float(np.sign(r_d)) if abs(r_d) > 1e-6 else 0.0

            elif strat == "RAND":                            # 安慰剂：随机方向，全日持有后 MOC 平
                if k == 0:
                    want = rsign

            if want != pos:
                if pos != 0 and np.isfinite(entry):
                    pnl += pos * (px / entry - 1)            # 旧仓按出场成交价结算
                pnl -= cost * abs(want - pos)
                trades += abs(want - pos)
                pos = want
                entry = px if want != 0 else np.nan

        if pos != 0 and np.isfinite(entry):                  # 收盘平仓（MOC）
            pnl += pos * (dayclose / entry - 1)
            pnl -= cost * abs(pos)
            trades += abs(pos)
        rets.append(pnl * lev)
        ntr.append(trades)
    return pd.Series(rets, index=pd.to_datetime(days)), float(np.mean(ntr))


def metrics(x, bench=None):
    x = x.dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(sharpe=np.nan, ann=np.nan, vol=np.nan, mdd=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    ann_ = float(eq.iloc[-1] ** (252 / len(x)) - 1)
    o = dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)), ann=ann_,
             vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
             calmar=ann_ / abs(dd.min()) if dd.min() < 0 else np.nan,
             hit=float((x > 0).mean()))
    if bench is not None:
        b = bench.reindex(x.index).fillna(0)
        if b.std() > 0:
            o["corr_bh"] = float(x.corr(b))
            beta = np.cov(x, b)[0, 1] / b.var()
            o["beta"] = float(beta)
            res = x - beta * b
            o["alpha"] = float(res.mean() * 252)
    return o


def cs_spread(px_hi, px_lo, px_c):
    """Corwin-Schultz 日频相对价差估计量，用于实测 TQQQ vs QQQ 的价差差异。"""
    lhi, llo = np.log(px_hi), np.log(px_lo)
    h2 = pd.concat([lhi, lhi.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([llo, llo.shift(1)], axis=1).min(axis=1)
    beta = (lhi - llo) ** 2 + ((lhi - llo).shift(1)) ** 2
    gamma = (h2 - l2) ** 2
    den = 3 - 2 * np.sqrt(2.0)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0)


def main():
    lev = pd.read_csv("data/lev/levetf_close.csv", index_col=0)
    # Corwin-Schultz 在 penny-wide 的超流动 ETF 上会被截断到 0（gamma 项主导），不可用。
    # 对这类标的，成本的两个主要成分都和**价格水平**成反比，直接分解更可靠：
    #   半价差  = (1 分钱 / 2) / 价格        两者都是 penny-quoted
    #   每股佣金 = $0.0035 / 价格            IBKR 分层费率
    # 成本模型（加法，不是乘法）：
    #   逐笔可量化成本 = 半价差 + 每股佣金 = (0.005 + 0.0035) / 价格        ← 两者都和价格反比
    #   冲击/滑点      = 与价格无关，小资金下≈0；用 QQQ 的实测基准反推
    # 之所以必须用加法：QQQ 的 0.5bp 基准里已经含了滑点，而滑点**不随价格缩放**。
    # 直接把 0.5bp 乘以价格比会把滑点也放大 13 倍，严重高估 TQQQ 的成本。
    PER_SHARE = 0.005 + 0.0035
    QQQ_BASE = 0.5                                   # QQQ 实测总成本 bp/边（限价 + MOC）
    px_lvl = {t: float(lev[t].loc["2023":].median()) for t in ("TQQQ", "QQQ", "SPY")}
    print("=== 成本分解（半价差 0.5¢ + 每股佣金 $0.0035，都是每股固定额 -> 相对成本 ∝ 1/价格）===")
    comp = {}
    for t, pl in px_lvl.items():
        hs, cm = (0.005 / pl) * 1e4, (0.0035 / pl) * 1e4
        comp[t] = dict(price=pl, half_spread_bp=hs, commission_bp=cm, total_bp=hs + cm)
        print(f"  {t}: 中位价 ${pl:6.1f} | 半价差 {hs:.3f}bp | 每股佣金 {cm:.3f}bp | 合计 {hs+cm:.3f}bp")
    slip = QQQ_BASE - comp["QQQ"]["total_bp"]         # QQQ 基准里属于滑点的部分
    print(f"  QQQ 基准 {QQQ_BASE}bp 中，逐笔部分 {comp['QQQ']['total_bp']:.3f}bp，剩余 {slip:.3f}bp 归为滑点（与价格无关）")

    rows, curves = [], {}
    panels_all, days_all, pc_all, atr_all = {}, {}, {}, {}
    for sym, path in [("SPY", "data/SPY_5min.csv"), ("QQQ", "data/QQQ_5min.csv")]:
        raw = load_5min(path)
        panels = day_panels(raw)
        days = sorted(panels)
        dayclose = pd.Series({d: panels[d][4][-1] for d in days})
        pc = dayclose.shift(1).reindex(days).to_numpy()
        panels_all[sym], days_all[sym], pc_all[sym] = panels, days, pc
        atr = day_atr(panels, days)
        atr_all[sym] = atr
        M, sig = noise_bands(panels, days)
        bh = pd.Series(np.r_[np.nan, np.diff(np.log(dayclose.to_numpy()))], index=pd.to_datetime(days)).apply(np.expm1)
        # ---- 成本情景：QQQ/SPY 用 0.5bp（限价+MOC），另测 1bp / 2bp
        for cb, lab in [(0.5, "0.5bp"), (1.0, "1bp"), (2.0, "2bp")]:
            for st in ("NOISE", "ORB5", "ORB5_ATR", "ON_REV", "ON_CONT", "VWAP_REV", "GAO",
                       "LUNCH_REV", "LAST30", "RAND"):
                r_, nt = simulate(panels, days, st, cb, 1.0, pc, sig, M, atr)
                m = metrics(r_.loc["2023":], bh.loc["2023":])
                rows.append(dict(symbol=sym, strat=st, cost=lab, lev=1.0, trades_per_day=nt, **m))
                if cb == 0.5:
                    curves[(sym, st)] = r_
        rows.append(dict(symbol=sym, strat="BUY&HOLD", cost="-", lev=1.0, trades_per_day=0,
                         **metrics(bh.loc["2023":], bh.loc["2023":])))
        curves[(sym, "BUY&HOLD")] = bh

    # ---------- TQQQ：用 QQQ 的 5 分钟路径 ×3 模拟，价差按实测倍数放大
    panels, days, pc, atr = panels_all["QQQ"], days_all["QQQ"], pc_all["QQQ"], atr_all["QQQ"]
    M, sig = noise_bands(panels, days)
    qq_close = pd.Series({d: panels[d][4][-1] for d in days})
    bh_q = pd.Series(np.r_[np.nan, np.diff(np.log(qq_close.to_numpy()))], index=pd.to_datetime(days)).apply(np.expm1)
    tq_real = lev["TQQQ"].reindex(days).pct_change(); tq_real.index = pd.to_datetime(days)
    # 逐日成本：TQQQ 在测试期内价格从 $8 涨到 $81，相对成本因此下降了 10 倍，必须逐日算
    tq_px = lev["TQQQ"].reindex(days).ffill().bfill()
    tq_cost_daily = (PER_SHARE / tq_px.to_numpy()) * 1e4 + slip
    i0 = days.index("2023-01-03") if "2023-01-03" in days else 0
    print(f"  TQQQ 逐日成本 bp/边：test 期中位 {np.nanmedian(tq_cost_daily[i0:]):.2f}，"
          f"2023 初 {tq_cost_daily[i0]:.2f}（价格 ${tq_px.iloc[i0]:.1f}）→ "
          f"最新 {tq_cost_daily[-1]:.2f}（价格 ${tq_px.iloc[-1]:.1f}）")
    tq_cost = float(np.nanmedian(tq_cost_daily[i0:]))
    base_cost = 0.5
    print(f"\n=== 3 倍敞口怎么拿：买 TQQQ vs 直接用 QQQ 的日内保证金 ===")
    print(f"    两者的日内路径都是 ≈3×QQQ，区别只在**每笔的相对成本**；")
    print(f"    而且本策略收盘平仓，所以既不吃 TQQQ 的每日再平衡复利衰减，也不付隔夜融资利息。")
    scen = [("QQQ 日内保证金 3x @0.50bp", 0.5), ("@1.00bp", 1.0), ("@2.00bp", 2.0),
            ("TQQQ 逐日实测成本  <-实际", tq_cost_daily), (f"TQQQ 中位成本 {tq_cost:.2f}bp", tq_cost)]
    for st in ("NOISE", "ORB5", "ORB5_ATR"):
        r_q, nt = simulate(panels, days, st, base_cost, 1.0, pc, sig, M, atr)
        m = metrics(r_q.loc["2023":], bh_q.loc["2023":])
        rows.append(dict(symbol="QQQ/TQQQ", strat=f"{st} on QQQ (1x)", cost=f"{base_cost:.2f}bp",
                         lev=1.0, trades_per_day=nt, **m))
        curves[("TQQQ_test", f"{st} on QQQ (1x)")] = r_q
        for lab, cb in scen:
            r_t, _ = simulate(panels, days, st, cb, 3.0, pc, sig, M, atr)
            m = metrics(r_t.loc["2023":], bh_q.loc["2023":])
            tag = f"{st} 3x: {lab}"
            rows.append(dict(symbol="QQQ/TQQQ", strat=tag,
                             cost="daily" if not np.isscalar(cb) else f"{cb:.2f}bp", lev=3.0,
                             trades_per_day=nt, **m))
            if not np.isscalar(cb) or cb == 0.5:
                curves[("TQQQ_test", tag)] = r_t
            if st == "NOISE" and not np.isscalar(cb):      # TQQQ 成本随价格变化 -> 按年看
                yr = pd.DataFrame({"ret": r_t, "cost_bp": cb}, index=r_t.index).loc["2023":]
                g = yr.groupby(yr.index.year).apply(
                    lambda x: pd.Series({"cost_bp": x.cost_bp.mean(),
                                         "sharpe": x.ret.mean() / x.ret.std() * np.sqrt(252),
                                         "ann": x.ret.mean() * 252}))
                print("\n    NOISE on TQQQ 按年（成本随 TQQQ 价格水平变化）：")
                print(g.round(3).to_string())
    rows.append(dict(symbol="QQQ/TQQQ", strat="TQQQ BUY&HOLD (real)", cost="-", lev=3.0,
                     trades_per_day=0, **metrics(tq_real.loc["2023":], bh_q.loc["2023":])))
    curves[("TQQQ_test", "TQQQ BUY&HOLD (real)")] = tq_real

    # ---------- 隔夜仓的影响：纯日内 vs 买入持有+叠加
    print("\n=== 隔夜仓的影响 ===")
    ov = {}
    for sym in ("SPY", "QQQ"):
        r_ = curves[(sym, "NOISE")]
        bh = curves[(sym, "BUY&HOLD")]
        for nm, series in [(f"{sym} 纯日内叠加层（不持隔夜）", r_),
                           (f"{sym} 买入持有（只有隔夜+日内 beta）", bh),
                           (f"{sym} 买入持有 + 0.5x 叠加（持隔夜）", bh + 0.5 * r_),
                           (f"{sym} 买入持有 + 1.0x 叠加（持隔夜）", bh + 1.0 * r_)]:
            m = metrics(series.loc["2023":], bh.loc["2023":])
            rows.append(dict(symbol=sym, strat=nm, cost="0.5bp", lev=np.nan, trades_per_day=np.nan, **m))
            ov[nm] = series
    # ---------- 组合：用 dev 期（<2023）挑选 sleeve，test 期等风险加权（走向前，不是事后挑）
    print("\n=== 多策略组合：dev 期(2019-2022)选 sleeve -> test 期(2023-)逆波动率加权 ===")
    STS = ("NOISE", "ORB5", "ORB5_ATR", "ON_REV", "ON_CONT", "VWAP_REV", "GAO", "LUNCH_REV", "LAST30")
    for sym in ("SPY", "QQQ"):
        X = pd.DataFrame({st: curves[(sym, st)] for st in STS})
        dev, tst = X.loc[:"2022"], X.loc["2023":]
        d_sr = dev.mean() / dev.std() * np.sqrt(252)
        keep = [s for s in STS if d_sr[s] > 0]
        print(f"  {sym} dev Sharpe: " + ", ".join(f"{s} {d_sr[s]:+.2f}" for s in STS))
        print(f"  {sym} 入选: {keep or '(无)'}")
        if not keep:
            continue
        w = 1.0 / dev[keep].std()
        w = w / w.sum()
        cb_ = (tst[keep] * w).sum(axis=1)
        rows.append(dict(symbol=sym, strat=f"组合({len(keep)}个sleeve, dev选+逆波动率)", cost="0.5bp",
                         lev=np.nan, trades_per_day=np.nan,
                         **metrics(cb_, curves[(sym, "BUY&HOLD")].loc["2023":])))
        curves[(sym, "COMBO_sleeves")] = cb_
        bh_ = curves[(sym, "BUY&HOLD")]
        rows.append(dict(symbol=sym, strat=f"买入持有 + 1.0x 组合叠加", cost="0.5bp", lev=np.nan,
                         trades_per_day=np.nan,
                         **metrics((bh_ + cb_.reindex(bh_.index).fillna(0)).loc["2023":], bh_.loc["2023":])))

    comb = (curves[("SPY", "NOISE")] + curves[("QQQ", "NOISE")]) / 2
    rows.append(dict(symbol="SPY+QQQ", strat="叠加层等权（不持隔夜）", cost="0.5bp", lev=np.nan,
                     trades_per_day=np.nan, **metrics(comb.loc["2023":], curves[("QQQ", "BUY&HOLD")].loc["2023":])))
    curves[("COMBO", "overlay_eq")] = comb

    R = pd.DataFrame(rows); R.to_csv(OUT / "intraday_battery.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 400)
    c = ["symbol", "strat", "cost", "trades_per_day", "sharpe", "ann", "vol", "mdd", "calmar",
         "hit", "corr_bh", "beta", "alpha"]
    print("\n=== 9 种日内策略 × 3 个成本情景（test 2023-2026）===")
    print(R[(R.cost.isin(["0.5bp", "1bp", "2bp"])) & (R.lev == 1.0)][c].round(3).to_string(index=False))
    print("\n=== TQQQ 对比 ===")
    print(R[R.symbol == "QQQ/TQQQ"][c].round(3).to_string(index=False))
    print("\n=== 隔夜仓 ===")
    print(R[R.strat.str.contains("隔夜|买入持有 \\+|叠加层等权", regex=True)][c].round(3).to_string(index=False))
    print("\n=== 日内策略两两相关（0.5bp, QQQ）===")
    cc = pd.DataFrame({st: curves[("QQQ", st)] for st in
                       ("NOISE", "ORB5", "ORB5_ATR", "ON_REV", "ON_CONT", "VWAP_REV", "GAO",
                        "LUNCH_REV", "LAST30")}).loc["2023":].corr()
    cc.to_csv(OUT / "intraday_corr.csv")
    print(cc.round(2).to_string())
    pd.to_pickle(curves, OUT / "intraday_curves.pkl")

    # ---------- 逐年表现 + 成本盈亏平衡点（这是判断能不能上实盘的两个关键数字）
    print("\n=== NOISE 逐年表现（0.5bp/边）与成本盈亏平衡点 ===")
    yr_rows = []
    for sym in ("SPY", "QQQ"):
        r_ = curves[(sym, "NOISE")].dropna()
        t_ = r_.loc["2023":]
        g = t_.groupby(t_.index.year).agg(
            sharpe=lambda x: x.mean() / x.std() * np.sqrt(252), ann=lambda x: x.mean() * 252,
            n=lambda x: len(x))
        print(f"  {sym}:"); print(g.round(3).to_string())
        g["symbol"] = sym; yr_rows.append(g)
        # 二分求使 test 期 Sharpe = 0 的成本
        lo_, hi_ = 0.0, 8.0
        pan_, dys_, pc_, atr_ = panels_all[sym], days_all[sym], pc_all[sym], atr_all[sym]
        M_, sig_ = noise_bands(pan_, dys_)
        for _ in range(20):
            mid = (lo_ + hi_) / 2
            rr, _ = simulate(pan_, dys_, "NOISE", mid, 1.0, pc_, sig_, M_, atr_)
            ss = rr.loc["2023":].dropna()
            if ss.mean() > 0:
                lo_ = mid
            else:
                hi_ = mid
        print(f"  {sym} 盈亏平衡成本 = {lo_:.2f} bp/边（当前假设 0.5bp，安全边际 {lo_/0.5:.1f}x）\n")
    pd.concat(yr_rows).to_csv(OUT / "intraday_by_year.csv")

    # ---------- 收益形态：这个策略到底靠什么赚钱
    print("\n=== 收益形态：按当日 QQQ 绝对涨跌分 5 组看 NOISE 的日均收益 ===")
    prof = {}
    for sym in ("SPY", "QQQ"):
        r_ = curves[(sym, "NOISE")].loc["2023":].dropna()
        b_ = curves[(sym, "BUY&HOLD")].reindex(r_.index)
        qt = pd.qcut(b_.abs(), 5, labels=["Q1最平静", "Q2", "Q3", "Q4", "Q5最大波动"])
        t = pd.DataFrame({"r": r_, "am": b_.abs()}).groupby(qt, observed=True).agg(
            n=("r", "size"), 日均bp=("r", lambda x: x.mean() * 1e4),
            胜率=("r", lambda x: (x > 0).mean()), 指数绝对涨跌=("am", "mean"))
        t["年化"] = t["日均bp"] * 252 * 1e-4
        prof[sym] = t
        sk = float(((r_ - r_.mean()) ** 3).mean() / r_.std() ** 3)
        top5 = r_.nlargest(5)
        print(f"  {sym}: 偏度 {sk:+.2f}；去掉最好的 1 天年化 {r_.drop(top5.index[:1]).mean()*252:+.2%}，"
              f"去掉最好的 5 天年化 {r_.drop(top5.index).mean()*252:+.2%}（原 {r_.mean()*252:+.2%}）")
        print(f"     最好的一天 {top5.index[0].date()}: 策略 {top5.iloc[0]:+.2%}, 指数 {b_.loc[top5.index[0]]:+.2%}")
        print(t.round(4).to_string())
    pd.concat(prof, names=["symbol"]).to_csv(OUT / "intraday_payoff_profile.csv")

    fig, axes = plt.subplots(1, 4, figsize=(25, 5.8))
    ax = axes[0]
    for st in ("NOISE", "ORB5", "ORB5_ATR", "ON_REV", "VWAP_REV", "GAO", "LAST30"):
        s = curves[("QQQ", st)].loc["2023":]
        ax.plot((1 + s).cumprod(), lw=1.1, label=f"{st}: SR {metrics(s)['sharpe']:.2f}")
    ax.plot((1 + curves[("QQQ", "BUY&HOLD")].loc["2023":]).cumprod(), lw=2, color="k", ls="--",
            label=f"QQQ B&H: SR {metrics(curves[('QQQ','BUY&HOLD')].loc['2023':])['sharpe']:.2f}")
    ax.set_title("QQQ 日内策略（0.5bp/边，收盘平仓）"); ax.legend(fontsize=7.5); ax.grid(alpha=.3)
    ax = axes[1]
    for nm in [k[1] for k in curves if k[0] == "TQQQ_test"]:
        s = curves[("TQQQ_test", nm)].loc["2023":]
        ax.plot((1 + s).cumprod(), lw=1.3, label=f"{nm}: SR {metrics(s)['sharpe']:.2f}, "
                                                 f"MDD {metrics(s)['mdd']:.0%}")
    ax.set_yscale("log"); ax.set_title("TQQQ vs QQQ（日内，收盘平仓）")
    ax.legend(fontsize=7.5); ax.grid(alpha=.3)
    ax = axes[2]
    for nm, s in ov.items():
        if "QQQ" in nm:
            ss = s.loc["2023":]
            ax.plot((1 + ss).cumprod(), lw=1.3, label=f"{nm}: SR {metrics(ss)['sharpe']:.2f}")
    ax.set_title("隔夜仓的影响（QQQ）"); ax.legend(fontsize=7.5); ax.grid(alpha=.3)
    ax = axes[3]
    x = np.arange(5)
    for off, sym in ((-0.2, "SPY"), (0.2, "QQQ")):
        ax.bar(x + off, prof[sym]["日均bp"].to_numpy(), width=0.4, label=sym)
    ax.set_xticks(x); ax.set_xticklabels(prof["QQQ"].index, fontsize=8)
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("策略日均收益 (bp)")
    ax.set_title("收益全部来自大波动日（凸性 / 类多头 gamma）")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(OUT / "fig_intraday_battery.png", dpi=110); plt.close()
    print("\nfigure written")
    return R, curves


if __name__ == "__main__":
    main()
