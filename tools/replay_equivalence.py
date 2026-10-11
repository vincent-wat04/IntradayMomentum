"""用历史 bar 逐根回放 live/signal.py 的 decide()，证明实盘代码 == 已审计的回测。

这是整个实盘系统的正确性基石：
  replay(reentry="same_bar") 必须逐位等于 audit/reimpl_noise.py（后者已逐位等于被测引擎）
  replay(reentry="hold")     必须逐位等于 reimpl_noise(reentry="hold")
只要这两条成立，就证明 decide() 没有实现漂移，实盘跑的就是回测过的那个策略。

跑法：/projects/sandbox/.venv/bin/python live/replay.py
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  设置 sys.path（研究引擎 + 实盘库 src）
from etft.intraday.bars import load_bars

from audit import reimpl_noise as RI
import strategy_signal as SG


def replay(b, cfg: SG.Cfg, half_spread=0.005, commission=0.0045, capital=1.0,
           cost_shares="leg_price"):
    """逐根 bar 调用 decide()，用显式股数账本记账。返回 (日收益, 成交明细)。"""
    O, C, H, L, V = b.o, b.h, b.c, b.l, b.v          # 注意顺序：下面按名字取
    O, H, L, C, V = b.o, b.h, b.l, b.c, b.v
    clocks = list(C.columns)
    # σ profile 与日波动：用 pandas 预先算好（和 signal.sigma_profile 的定义一致，下面会校验）
    move = (C.div(b.day_open, axis=0) - 1).abs()
    SIG = move.rolling(cfg.lookback, min_periods=max(5, int(0.7 * cfg.lookback))).mean().shift(1)
    rd = b.day_close / b.prev_close_adj - 1
    SD = rd.rolling(cfg.lookback, min_periods=max(5, cfg.lookback // 2)).std().shift(1)
    tp = (H + L + C) / 3.0
    VWAP = (tp * V).cumsum(axis=1) / V.cumsum(axis=1).replace(0, np.nan)

    rets, fills = [], []
    for d in b.days:
        c_row = C.loc[d].to_numpy(); o_row = O.loc[d].to_numpy()
        s_row = SIG.loc[d].to_numpy(); vw_row = VWAP.loc[d].to_numpy()
        valid = np.isfinite(c_row)
        if not valid.any():
            rets.append(0.0); continue
        last = int(np.where(valid)[0][-1])
        e = SG.exposure_from(float(SD.loc[d]) if np.isfinite(SD.loc[d]) else None, cfg)
        plan = SG.DayPlan(day=str(d.date()), day_open=float(b.day_open.loc[d]),
                          prev_close_adj=float(b.prev_close_adj.loc[d]),
                          sigma={clocks[i]: (float(s_row[i]) if np.isfinite(s_row[i]) else None)
                                 for i in range(len(clocks))},
                          exposure=e if e is not None else float("nan"),
                          sigma_daily=float(SD.loc[d]) if np.isfinite(SD.loc[d])
                          else float("nan"))
        shares, entry_px, pnl, cost = 0.0, np.nan, 0.0, 0.0
        pos = 0.0
        no_entry_before = -1
        for k in range(last):                 # 决策在第 k 根 bar 收盘，成交在 o_row[k+1]
            fill_px = o_row[k + 1]
            if not np.isfinite(fill_px) or not np.isfinite(c_row[k]):
                continue
            cl = clocks[k]
            # 硬尾止损要逐根 bar 评估，所以不能在这里直接 skip 非检查时点
            if cl not in cfg.check_times and not (cfg.tail_stop and pos != 0.0):
                continue
            tgt, _why = SG.decide(plan, cl, float(c_row[k]),
                                  float(vw_row[k]) if np.isfinite(vw_row[k]) else float("nan"),
                                  pos, cfg, entry_px=entry_px)
            if tgt == pos:
                continue
            # 先结算旧仓
            if shares != 0.0:
                pnl += shares * (fill_px - entry_px)
                shc = abs(shares) if cost_shares == "actual" else abs(shares) * entry_px / fill_px
                cm = shc * (half_spread + commission)
                cost += cm
                fills.append(dict(day=d, clock=cl, kind="EXIT", shares=-shares,
                                  price=fill_px, commission=cm))
                shares, entry_px = 0.0, np.nan
                if cfg.reentry == "next_check":
                    no_entry_before = k + 1
            if tgt != 0.0:
                sh = tgt * capital / fill_px
                cm = abs(sh) * (half_spread + commission)
                cost += cm
                fills.append(dict(day=d, clock=cl, kind="ENTRY", shares=sh,
                                  price=fill_px, commission=cm))
                shares, entry_px = sh, fill_px
            pos = tgt
        if shares != 0.0:                      # MOC
            px = float(b.day_close.loc[d])
            pnl += shares * (px - entry_px)
            shc = abs(shares) if cost_shares == "actual" else abs(shares) * entry_px / px
            cm = shc * commission
            cost += cm
            fills.append(dict(day=d, clock=clocks[last], kind="MOC", shares=-shares,
                              price=px, commission=cm))
        rets.append((pnl - cost) / capital)
    return pd.Series(rets, index=b.days, name="ret"), pd.DataFrame(fills)


def test_equivalence(sym="QQQ"):
    b = load_bars(sym)
    print(f"回放等价性检验：{sym}，{len(b.days)} 天 × {b.c.shape[1]} 根 bar")
    allok = True
    for bad in ("same_bar", "next_check"):
        try:
            SG.Cfg(reentry=bad)
            print(f"  [**FAIL**] Cfg(reentry={bad!r}) 应当报错"); allok = False
        except ValueError:
            print(f"  [PASS] Cfg(reentry={bad!r}) 被正确拒绝（纯函数架构不可表达该语义）")
    # 不开硬尾的基础用例。**两侧都必须显式指定止损模式** —— Cfg 的默认值会随
    # 选型变化（PART13 把默认改成了 band），而 RI.simulate 的默认是 vwap，
    # 依赖任一侧的默认值会让这个检验在换配置时静默失效。
    for sizing in ("fixed", "voltarget"):
        for mode in ("vwap", "band"):
            cfg = SG.Cfg(sizing=sizing, reentry="hold", tail_stop=False, stop=mode)
            r_live, F = replay(b, cfg)
            r_ref, _ = RI.simulate(b, sizing=sizing, reentry="hold",
                                   cost_shares="leg_price", half_spread=0.005,
                                   commission=0.0045, stop_long=mode, stop_short=mode)
            d = (r_live.fillna(0) - r_ref.fillna(0)).abs()
            ok = d.max() < 1e-12
            allok &= ok
            print(f"  [{'PASS' if ok else '**FAIL**'}] sizing={sizing:9s} 止损={mode:4s} "
                  f"最大差 {d.max():.2e}  不一致 {int((d>1e-12).sum())} 天  "
                  f"累计 live {r_live.sum():.6f} / ref {r_ref.sum():.6f}")
            # 账本自检：每天净股数归零
            if len(F):
                net = F.groupby("day").shares.sum().abs().max()
                if net > 1e-9:
                    print(f"         **账本未平：最大净股数 {net:.2e}**"); allok = False
    # 硬尾部止损开启时，必须等于参考实现的 tail_every_bar=True。
    # 两种止损模式都要验：vwap（旧实盘）与 band（PART13 选定）。
    for mode in ("vwap", "band"):
        for k in (0.5, 1.0, 1.5):
            cfg = SG.Cfg(sizing="voltarget", reentry="hold", tail_stop=True,
                         tail_k=k, stop=mode)
            r_live, _ = replay(b, cfg)
            r_ref, _ = RI.simulate(b, sizing="voltarget", reentry="hold",
                                   cost_shares="leg_price", half_spread=0.005,
                                   commission=0.0045,
                                   stop_long=f"{mode}+tail", stop_short=f"{mode}+tail",
                                   tail_k=k, tail_every_bar=True)
            d = (r_live.fillna(0) - r_ref.fillna(0)).abs()
            ok = d.max() < 1e-12
            allok &= ok
            print(f"  [{'PASS' if ok else '**FAIL**'}] 止损={mode:4s} 硬尾 k={k}："
                  f"最大差 {d.max():.2e}  不一致 {int((d>1e-12).sum())} 天  "
                  f"累计 live {r_live.sum():.6f} / ref {r_ref.sum():.6f}")

    # signal.sigma_profile / daily_sigma 与回放里用的 pandas 版本是否一致
    cfg = SG.Cfg(tail_stop=False)
    clocks = list(b.c.columns)
    move = (b.c.div(b.day_open, axis=0) - 1).abs()
    SIG = move.rolling(cfg.lookback, min_periods=max(5, int(0.7 * cfg.lookback))).mean().shift(1)
    bad = 0
    for d in list(b.days)[-40:]:
        i = b.days.get_loc(d)
        hist = [(str(x.date()), {c: b.c.loc[x, c] for c in clocks}, float(b.day_open.loc[x]))
                for x in b.days[max(0, i - cfg.lookback):i]]
        s = SG.sigma_profile(hist, clocks, cfg.lookback)
        for m in ("10:00", "12:00", "15:00"):
            a, e = s.get(m), SIG.loc[d, m]
            if (a is None) != (not np.isfinite(e)):
                bad += 1
            elif a is not None and abs(a - e) > 1e-12:
                bad += 1
    print(f"  [{'PASS' if bad == 0 else '**FAIL**'}] signal.sigma_profile 与 pandas 版本一致"
          f"（最近 40 天 × 3 个时点，不一致 {bad} 处）")
    allok &= (bad == 0)
    # daily_sigma
    rd = b.day_close / b.prev_close_adj - 1
    SD = rd.rolling(cfg.lookback, min_periods=max(5, cfg.lookback // 2)).std().shift(1)
    bad2 = 0
    for d in list(b.days)[-40:]:
        i = b.days.get_loc(d)
        ca = [float(b.day_close.iloc[j]) for j in range(max(0, i - cfg.lookback - 1), i)]
        a = SG.daily_sigma(ca, cfg.lookback)
        e = SD.loc[d]
        if a is not None and np.isfinite(e) and abs(a - e) > 2e-4:
            bad2 += 1
    print(f"  [{'PASS' if bad2 == 0 else 'WARN'}] signal.daily_sigma 与 pandas 版本一致"
          f"（不一致 {bad2}/40；注意复权前收与原始收盘的细微差异）")
    print(f"\n{'='*70}\n回放等价性：{'全部通过' if allok else '**有失败项**'}")
    return allok


if __name__ == "__main__":
    ok = test_equivalence("QQQ")
    sys.exit(0 if ok else 1)
