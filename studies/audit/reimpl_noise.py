"""噪声区域策略的**独立重写实现**：显式股数/现金账本的事件驱动模拟器。

为什么要重写一遍：读代码找不出记账错误。两个**结构不同**的实现如果能对到 1e-10，
那么两者同时错成一样的概率极低。已验证引擎 (etft/intraday/strategy.py) 用的是
"入场价→出场价的比值"记账；这里用的是"显式股数 + 现金 + 逐笔佣金"的账本，
从论文描述重新推导，刻意不看原实现的内部结构。

论文规则（Zarattini, Aziz & Barbon 2024）：
  σ_{t,m} = 过去 L=14 个交易日里，同一时点 m 的 |close_{d,m}/open_d − 1| 的均值
  UB_{t,m} = max(open_t, prev_close_adj_t) × (1 + vm·σ_{t,m})
  LB_{t,m} = min(open_t, prev_close_adj_t) × (1 − vm·σ_{t,m})
  在检查时点（10:00, 10:30, ..., 15:00 的 bar 收盘）：
      空仓时   close > UB -> 做多 ；close < LB -> 做空
      持仓时   多头止损 = max(UB, VWAP)，空头止损 = min(LB, VWAP)
  仓位规模 = min(max_lev, target_vol / σ_daily_14d)
  决策在 bar 收盘，成交在下一根 bar 开盘；收盘 MOC 平仓。

账本约定（和被测引擎保持一致，以便对账）：
  每条腿的名义金额 = pos × 初始资金 C（**不做日内复利**），
  所以一笔从 p1 到 p2 的持仓，损益 = pos × C × (p2/p1 − 1)。
  成本：主动单付 (半价差 + 佣金)/股；MOC 单只付佣金/股（引擎口径）。
  日收益 = 当日损益合计 / C。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CHECKS = [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)][:-1]   # 10:00..15:00


@dataclass
class Fill:
    day: pd.Timestamp
    clock: str
    kind: str          # ENTRY_LONG / ENTRY_SHORT / STOP / MOC
    shares: float      # 带符号的股数变化
    price: float
    commission: float


@dataclass
class Ledger:
    capital: float = 1.0
    fills: list = field(default_factory=list)


def sigma_profile(close, day_open, lookback=14):
    """过去 lookback 天同一时点的平均 |close/open − 1|，严格只用过去（shift(1)）。"""
    move = (close.div(day_open, axis=0) - 1).abs()
    return move.rolling(lookback, min_periods=max(5, int(0.7 * lookback))).mean().shift(1)


def daily_sigma(day_close, prev_close_adj, lookback=14):
    r = day_close / prev_close_adj - 1
    return r.rolling(lookback, min_periods=max(5, lookback // 2)).std().shift(1)


def session_vwap(h, l, c, v):
    tp = (h + l + c) / 3.0
    return (tp * v).cumsum(axis=1) / v.cumsum(axis=1).replace(0, np.nan)


def _tp_applies(tp_side, shares):
    """止盈是否作用于当前这一侧。"""
    if tp_side == "both":
        return True
    return (tp_side == "long" and shares > 0) or (tp_side == "short" and shares < 0)


def simulate(b, vm=1.0, lookback=14, target_vol=0.02, max_lev=4.0, sizing="voltarget",
             half_spread=0.005, commission=0.0045, capital=1.0, checks=None,
             stop="vwap", reentry="same_bar", cost_shares="leg_price",
             stop_long=None, stop_short=None, vm_long=None, vm_short=None,
             tail_k=0.5, giveback=0.5, tail_every_bar=False,
             tp_k=None, tp_side="both", tp_every_bar=False, tp_rearm="done",
             tp_rand_p=None, tp_rand_seed=0, short_scale=1.0, tp_days=None,
             sides="both", skip_dows=(), skip_days=None):
    """返回 (日收益 Series, 成交明细 DataFrame)。

    reentry —— 止损与开仓条件在同一根 bar 同时成立时怎么处理（论文没写清楚，是语义歧义）：
      "same_bar"   先平后立刻重开（被测引擎的行为）。如果重开的是同方向，
                   净仓位没变，却付了 2 条腿的成本。
      "next_check" 平掉后这根 bar 不再开仓，等下一个检查时点（30 分钟后）。
      "hold"       如果同方向的开仓条件仍成立，就**不触发止损**（继续持有，
                   不付那 2 条腿的冤枉成本）。实盘应该用这个。
    stop_long / stop_short —— 每侧的止损方式，可取：
        "vwap"       多头 max(UB, 当日累计VWAP)、空头 min(LB, VWAP)（论文的改进版，最紧）
        "band"       多头跌破 LB、空头升破 UB（要穿过整条带，较松）
        "none"       完全不止损（也就无法反手）
        "tail"       **只切尾部**：相对入场价的不利偏离超过 tail_k × σ_daily 才出场
        "band+tail"  band 或 tail 任一触发
        "vwap+tail"  vwap 或 tail 任一触发
        "giveback"   从最大浮盈回吐 giveback 比例就出场（跟踪止盈）
        "tail+giveback"  两者任一触发
      为什么用 σ_daily 做刻度：仓位本身是 min(4, 2%/σ_daily)，所以把止损放在
      k×σ_daily 处 -> 单笔最大亏损 ≈ 敞口 × k × σ_daily = 2%·k 的**资金**，
      与波动率环境无关。k=0.5 就是"单笔最多亏 1% 资金"。
        注意 "none" 不只是"不止损"：反手是走止损路径实现的，所以关掉止损后
        仓位也无法反手，策略退化成"当日第一个信号持到收盘"。换手会显著下降。
        诊断发现空头 64.6% 被止损打掉、且止损交易胜率只有 10.1%，而持到收盘的空头
        胜率 77.6%/+59.2bp —— 所以"空头止损是否太紧"是个可检验的假设。
    vm_long / vm_short —— 两侧分别的噪声带宽度倍数（默认都跟随 vm）。
    tp_k / tp_side / tp_every_bar / tp_rearm —— **真止盈**（默认 tp_k=None 完全关闭）。
      为什么要单独开一个槽、而不是复用 giveback：giveback 挂在止损槽上，会被
      reentry="hold" 的那句 `side_now == sign(shares) -> hit=False` 抑制掉 ——
      也就是"趋势还在、浮盈还在涨"的时候恰好不触发，那不是止盈。
      本槽的触发**不受 reentry 语义干预**。
      tp_k    有利偏离 > tp_k × σ_daily 就平仓。与 tail_k 同一把尺子：
              敞口 = min(4, 2%/σ_daily)，所以 tp_k 的止盈 ≈ 锁定 2%·tp_k 的资金。
      tp_side "short" 只对空头止盈 / "long" 只对多头 / "both" 两侧。
              诊断（takeprofit_diag.py）：空头放弃段 dev −12.7~−30.1bp、
              test −14.8~−17.6bp（按天聚类 t≈−2.3~−3.6，同号同量级）；
              多头 dev 负但 test≈0 且高阈值翻正 -> 只做空头侧。
      tp_rearm 止盈后当天怎么再进场（决定了止盈到底是"离场"还是"换手"）：
              "done"       当天不再开仓（止盈即收工）
              "next_check" 下一个检查时点条件仍成立就再开（可同向多次开仓）
              "rearm"      必须先等价格**回到带内**（信号归零）再次突破才允许开仓
      注意 tp_rearm="next_check" 若不配合别的约束，会在下一个检查点立刻原向重开，
      等于白付 2 条腿 —— 这正是需要实测的地方。
    tp_rand_p / short_scale —— **安慰剂对照**，用来回答"止盈的收益到底来自哪里"。
      空头止盈天然会缩短空头的在场时间，而"少做空"本身就已知有利
      （long-only 独立 Sharpe 1.185 > 多空 0.873）。所以必须把两件事分开：
      tp_rand_p    每个检查时点以固定概率 p **无条件**平掉空头（不看盈亏）。
                   把 p 调到与真止盈相同的离场频率：若 Sharpe 改善一样，
                   说明止盈没有择时含量，赚的只是"空头在场时间变短"。
      short_scale  空头仓位规模乘以这个系数（纯敞口缩放，不改变任何时点）。
                   这是第二个对照：匹配"空头敞口×时间"后还剩多少。
    sides —— "both"（多空，论文原版）/ "long"（只做多）/ "short"（只做空）。
      注意 long-only 的独立 Sharpe 更高但**作为 B&H 叠加层更差**
      （PART9：独立 1.185 vs 多空 0.873，但叠加贡献 +0.127 vs +0.268），
      因为空头腿正是这个 sleeve 的对冲性来源。
    skip_days —— 不交易的**具体日期**集合（用于"按某个日频条件剔除整天"的实验，
      例如规避极高 GEX 日）。与 skip_dows 同样处理：当日收益记 0、无成交。
    skip_dows —— 不交易的星期（0=周一 .. 4=周五），与实盘的 SKIP_WEEKDAYS 对应。
      被跳过的日子当日收益记 0、且不产生任何成交（所以逐笔统计也干净）。
      策略日内不留仓，所以"跳过某天"与"把该天收益置零"等价。
    cost_shares —— 出场腿的成本按哪个股数算：
      "leg_price"  用出场价反算股数（被测引擎的做法，严格说是错的）
      "actual"     用入场时真实买到的股数（正确的账本做法）
    """
    checks = set(checks or CHECKS)
    stop_long = stop_long or stop
    stop_short = stop_short or stop
    vm_l = vm if vm_long is None else vm_long
    vm_s = vm if vm_short is None else vm_short
    O, H, L, C, V = b.o, b.h, b.l, b.c, b.v
    sig = sigma_profile(C, b.day_open, lookback)
    vw = session_vwap(H, L, C, V)
    sd = daily_sigma(b.day_close, b.prev_close_adj, lookback)
    expo = (np.minimum(max_lev, target_vol / sd) if sizing == "voltarget"
            else pd.Series(1.0, index=b.days))
    times = list(C.columns)
    hi_ref = np.maximum(b.day_open, b.prev_close_adj.fillna(b.day_open))
    lo_ref = np.minimum(b.day_open, b.prev_close_adj.fillna(b.day_open))

    rets, fills = [], []
    skip = set(int(x) for x in skip_dows)
    skipd = set(pd.Timestamp(x) for x in (skip_days or ()))
    for d in b.days:
        if (skip and pd.Timestamp(d).dayofweek in skip) or \
                (skipd and pd.Timestamp(d) in skipd):
            rets.append(0.0); continue
        c_row, o_row = C.loc[d].to_numpy(), O.loc[d].to_numpy()
        s_row, vw_row = sig.loc[d].to_numpy(), vw.loc[d].to_numpy()
        valid = np.isfinite(c_row)
        if not valid.any():
            rets.append(0.0); continue
        last = int(np.where(valid)[0][-1])
        e = float(expo.loc[d]) if np.isfinite(expo.loc[d]) else np.nan
        sd_d = float(sd.loc[d]) if np.isfinite(sd.loc[d]) else np.nan
        ub = hi_ref.loc[d] * (1 + vm_l * s_row)      # 上轨用多头的 vm
        lb = lo_ref.loc[d] * (1 - vm_s * s_row)      # 下轨用空头的 vm
        if not np.isfinite(e) or not np.isfinite(s_row).any():
            rets.append(0.0); continue

        shares = 0.0            # 当前持有股数（带符号）
        entry_px = np.nan
        pnl = 0.0               # 当日累计损益（金额）
        cost = 0.0
        mfe = 0.0               # 本笔的最大浮盈（比例），giveback 止盈用
        no_exit_before = -1     # 开仓后，这根 bar 不允许再评估止损
        no_entry_before = -1    # next_check 语义下用
        need_rearm = False      # tp_rearm="rearm"：等价格回带内才解锁
        _rg = np.random.default_rng(tp_rand_seed + int(hash(str(d)) % 100000))
        for k in range(last):   # 决策在第 k 根 bar 收盘，成交在 o_row[k+1]
            fill_px = o_row[k + 1]
            if not np.isfinite(fill_px) or not np.isfinite(ub[k]):
                continue
            px_now = c_row[k]
            is_check = times[k] in checks
            # 非检查时点：硬尾部止损与止盈都要逐根 bar 评估
            if not is_check:
                if shares != 0 and k >= no_exit_before and np.isfinite(sd_d) \
                        and sd_d > 0 and np.isfinite(entry_px):
                    sgn_ = 1.0 if shares > 0 else -1.0
                    tail_hit = (tail_every_bar
                                and -sgn_ * (px_now / entry_px - 1.0) > tail_k * sd_d)
                    tp_hit = (tp_every_bar and tp_k is not None
                              and (tp_days is None or d in tp_days)
                              and _tp_applies(tp_side, shares)
                              and sgn_ * (px_now / entry_px - 1.0) > tp_k * sd_d)
                    if tail_hit or tp_hit:
                        kind = "TAILSTOP" if tail_hit else "TP"
                        pnl += shares * (fill_px - entry_px)
                        sh_cost = abs(shares) if cost_shares == "actual" else \
                            abs(shares) * entry_px / fill_px
                        cm = sh_cost * (half_spread + commission)
                        cost += cm
                        fills.append(Fill(d, times[k], kind, -shares, fill_px, cm))
                        shares = 0.0; entry_px = np.nan; mfe = 0.0
                        if tail_hit and reentry == "next_check":
                            no_entry_before = k + 1
                        if tp_hit:
                            if tp_rearm == "done":
                                no_entry_before = last + 1
                            elif tp_rearm == "next_check":
                                no_entry_before = k + 1
                            else:
                                need_rearm = True
                continue
            side_now = 1 if px_now > ub[k] else (-1 if px_now < lb[k] else 0)
            if (sides == "long" and side_now < 0) or (sides == "short" and side_now > 0):
                side_now = 0          # 该侧不交易：信号直接视为无

            # ---- 止损
            if shares != 0 and k >= no_exit_before:
                sl_mode = stop_long if shares > 0 else stop_short
                sgn_ = 1.0 if shares > 0 else -1.0
                # 相对入场价的不利偏离 / 有利偏离（比例）
                adverse = -sgn_ * (px_now / entry_px - 1.0)
                favor = sgn_ * (px_now / entry_px - 1.0)
                mfe = max(mfe, favor)
                hit = False
                if "vwap" in sl_mode:
                    sl = max(ub[k], vw_row[k]) if np.isfinite(vw_row[k]) else ub[k]
                    ss = min(lb[k], vw_row[k]) if np.isfinite(vw_row[k]) else lb[k]
                    hit = hit or (shares > 0 and px_now < sl) or (shares < 0 and px_now > ss)
                if "band" in sl_mode:
                    hit = hit or (shares > 0 and px_now < lb[k]) or (shares < 0 and px_now > ub[k])
                if "tail" in sl_mode and np.isfinite(sd_d) and sd_d > 0:
                    hit = hit or (adverse > tail_k * sd_d)
                if "giveback" in sl_mode and mfe > 0:
                    hit = hit or ((mfe - favor) > giveback * mfe and favor < mfe)
                if hit and reentry == "hold" and side_now == np.sign(shares):
                    hit = False          # 同方向开仓条件仍成立 -> 不平仓，省掉 2 条腿成本
                if hit:
                    pnl += shares * (fill_px - entry_px)
                    sh_cost = abs(shares) if cost_shares == "actual" else \
                        abs(shares) * entry_px / fill_px
                    cm = sh_cost * (half_spread + commission)
                    cost += cm
                    fills.append(Fill(d, times[k], "STOP", -shares, fill_px, cm))
                    shares = 0.0; entry_px = np.nan
                    if reentry == "next_check":
                        no_entry_before = k + 1
            # ---- 安慰剂：按固定概率无条件平掉空头（与盈亏无关）
            if shares < 0 and tp_rand_p is not None and k >= no_exit_before \
                    and _rg.random() < tp_rand_p:
                pnl += shares * (fill_px - entry_px)
                sh_cost = abs(shares) if cost_shares == "actual" else \
                    abs(shares) * entry_px / fill_px
                cm = sh_cost * (half_spread + commission)
                cost += cm
                fills.append(Fill(d, times[k], "RANDEXIT", -shares, fill_px, cm))
                shares = 0.0; entry_px = np.nan; mfe = 0.0
                if tp_rearm == "done":
                    no_entry_before = last + 1
                elif tp_rearm == "next_check":
                    no_entry_before = k + 1
                else:
                    need_rearm = True

            # ---- 止盈（检查时点；**不受 reentry 语义抑制**）
            if shares != 0 and tp_k is not None and k >= no_exit_before \
                    and (tp_days is None or d in tp_days) \
                    and _tp_applies(tp_side, shares) and np.isfinite(sd_d) \
                    and sd_d > 0 and np.isfinite(entry_px):
                sgn_ = 1.0 if shares > 0 else -1.0
                if sgn_ * (px_now / entry_px - 1.0) > tp_k * sd_d:
                    pnl += shares * (fill_px - entry_px)
                    sh_cost = abs(shares) if cost_shares == "actual" else \
                        abs(shares) * entry_px / fill_px
                    cm = sh_cost * (half_spread + commission)
                    cost += cm
                    fills.append(Fill(d, times[k], "TP", -shares, fill_px, cm))
                    shares = 0.0; entry_px = np.nan; mfe = 0.0
                    if tp_rearm == "done":
                        no_entry_before = last + 1
                    elif tp_rearm == "next_check":
                        no_entry_before = k + 1
                    else:
                        need_rearm = True

            # rearm 语义：价格回到带内（信号归零）才解锁下一次开仓
            if need_rearm and shares == 0 and side_now == 0:
                need_rearm = False

            # ---- 开仓（只在检查时点、且空仓）
            if shares == 0 and not need_rearm and k >= no_entry_before and side_now != 0:
                sh = side_now * (e * capital) / fill_px
                if side_now < 0:
                    sh *= short_scale
                cm = abs(sh) * (half_spread + commission)
                cost += cm
                fills.append(Fill(d, times[k], "ENTRY_LONG" if side_now > 0 else "ENTRY_SHORT",
                                  sh, fill_px, cm))
                shares = sh; entry_px = fill_px
                no_exit_before = k + 1
        # ---- 收盘 MOC 平仓
        if shares != 0:
            px = float(b.day_close.loc[d])
            pnl += shares * (px - entry_px)
            sh_cost = abs(shares) if cost_shares == "actual" else abs(shares) * entry_px / px
            cm = sh_cost * commission               # MOC 只付佣金，不付半价差
            cost += cm
            fills.append(Fill(d, times[last], "MOC", -shares, px, cm))
            shares = 0.0
        rets.append((pnl - cost) / capital)
    R = pd.Series(rets, index=b.days, name="ret")
    F = pd.DataFrame([f.__dict__ for f in fills])
    return R, F


def ledger_check(F, capital=1.0):
    """账本自检：每天净股数必须归零（日内策略不持隔夜）。"""
    if len(F) == 0:
        return True, "no fills"
    net = F.groupby("day").shares.sum()
    bad = net[net.abs() > 1e-9]
    return bad.empty, (f"{len(bad)} 天收盘后股数不为 0" if len(bad) else "每天收盘股数均为 0")
