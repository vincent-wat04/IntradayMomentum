"""用反向 ETF 实现空头腿：SQQQ(-3x) / QID(-2x) / PSQ(-1x) vs 直接做空 TQQQ。

动机：策略是多空双向，空头腿目前靠**做空 TQQQ**实现，这需要券源（locate/borrow）。
买入反向 ETF 则是**多头仓位，完全不需要借券**，也不受做空规则限制。

但要付两个代价，都必须实测而不是假设：
  1. **相对成本**：每股固定成本（半价差 + 佣金）÷ 价格。反向 ETF 因为长期跟踪下跌的
     倒数，价格会被反复反向拆股压得很低，所以相对成本高。
  2. **跟踪质量**：日内是否真的是 −3x / −2x。必须测 realized beta，不能信招牌。

另外注意：这些 ETF 的"每日再平衡复利衰减"对本策略**不适用**（收盘平仓，不跨日），
和 PART9 §1.1 对 TQQQ 的论证完全相同。
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars
from etft.intraday.strategy import NoiseCfg, run, buy_hold

OUT = Path("outputs/full/intraday"); OUT.mkdir(parents=True, exist_ok=True)
PER_SHARE = 0.005 + 0.0045
TEST, ALL = ("2023", "2026"), ("2019", "2026")
SOTA = dict(stop="vwap", sizing="voltarget", max_lev=4.0)
# 名义杠杆（招牌值）；realized 值下面实测
NOMINAL = {"QQQ": 1.0, "TQQQ": 3.0, "QLD": 2.0, "SQQQ": -3.0, "QID": -2.0, "PSQ": -1.0}


def real_price(sym):
    """真实挂牌价（把未来拆股/反向拆股还原回去）+ 复权收盘（算收益用）+ 成交量。"""
    h = yf.download(sym, start="2018-10-01", end="2026-10-05", auto_adjust=False,
                    progress=False, threads=False)
    h.columns = [c[0] if isinstance(c, tuple) else c for c in h.columns]
    h.index = pd.to_datetime(h.index).tz_localize(None)
    sp = yf.Ticker(sym).splits
    f = pd.Series(1.0, index=h.index)
    if len(sp):
        sp.index = pd.to_datetime(sp.index).tz_localize(None)
        for d, r in sp.items():
            f[h.index < d] *= r
    return pd.DataFrame(dict(listed=h.Close * f, adj=h["Adj Close"],
                             vol=h.Volume, dollar_vol=h.Volume * h.Close))


def chk(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def met(x, bench=None):
    x = pd.Series(x).dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(sharpe=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    a = float(eq.iloc[-1] ** (252 / len(x)) - 1)
    o = dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)), ann=a,
             vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
             calmar=a / abs(dd.min()) if dd.min() < 0 else np.nan)
    if bench is not None:
        b = pd.Series(bench).reindex(x.index).fillna(0)
        if b.std() > 0:
            o["beta"] = float(np.cov(x, b)[0, 1] / b.var())
            o["alpha"] = float((x - o["beta"] * b).mean() * 252)
    return o


def main():
    P = {s: real_price(s) for s in NOMINAL}
    b = load_bars("QQQ"); days = b.days; bh = buy_hold(b)

    # ---------- 1. 价格水平、流动性、相对成本
    print("=" * 108)
    print("1. 反向/杠杆 ETF 的价格水平与每单位 QQQ 敞口的成本（2023+ 中位）")
    print("=" * 108)
    rows = []
    for s, lev in NOMINAL.items():
        d = P[s].loc["2023":]
        px = float(d.listed.median())
        dv = float(d.dollar_vol.median())
        c_not = PER_SHARE / px * 1e4            # bp of 本 ETF 名义金额
        c_exp = c_not / abs(lev)                # bp of 每单位 QQQ 等价敞口
        rows.append(dict(代码=s, 名义杠杆=lev, 中位挂牌价=px,
                         日均成交额_亿美元=dv / 1e8, 名义成本bp=c_not, 单位敞口成本bp=c_exp,
                         拆股次数=len(yf.Ticker(s).splits)))
    C = pd.DataFrame(rows)
    print(C.round(3).to_string(index=False))

    # ---------- 2. realized beta（日频，2023+）：招牌值对不对
    print("\n" + "=" * 108)
    print("2. realized beta（对 QQQ 日收益回归，2023+）—— 不信招牌，实测")
    print("=" * 108)
    rq = P["QQQ"].adj.pct_change()
    rows2 = []
    for s, lev in NOMINAL.items():
        r = P[s].adj.pct_change()
        df = pd.concat([r.rename("y"), rq.rename("x")], axis=1).loc["2023":].dropna()
        beta = np.cov(df.y, df.x)[0, 1] / df.x.var()
        corr = df.y.corr(df.x)
        # 年化跟踪拖累：实际 vs 名义杠杆×QQQ
        drag = (df.y.mean() - lev * df.x.mean()) * 252
        rows2.append(dict(代码=s, 名义=lev, realized_beta=beta, 相关=corr,
                          年化跟踪拖累=drag, 偏离名义=beta - lev))
    B = pd.DataFrame(rows2)
    print(B.round(4).to_string(index=False))
    print("  注：跟踪拖累来自每日再平衡复利，对**收盘平仓**的日内策略不适用（PART9 §1.1）")

    # ---------- 3. 把空头腿换成反向 ETF 后，策略的净收益
    print("\n" + "=" * 108)
    print("3. 多空双向策略：空头腿用不同工具实现（30 相位均值）")
    print("=" * 108)
    # 逐相位跑，拿到 gross 与"多头换手/空头换手"的拆分
    # 引擎的 cost 列是总换手；需要按多空分开计成本 -> 用 record_trades 重建
    per_phase = {}
    for off in range(30):
        r = run(b, NoiseCfg(check_times=chk(off), cost_bp=1.0, record_trades=True, **SOTA))
        T = r.attrs["trades"]
        # 每天的多头/空头换手（进+出各一次，所以 ×2）
        T["day"] = pd.to_datetime(T.day)
        tl = T[T.dir > 0].groupby("day")["size"].sum().reindex(days).fillna(0) * 2
        ts = T[T.dir < 0].groupby("day")["size"].sum().reindex(days).fillna(0) * 2
        per_phase[off] = dict(gross=r["overlay_gross"], turn_l=tl, turn_s=ts,
                             turn_all=r["cost"] * 1e4)
    # 校验：多头+空头换手 ≈ 引擎总换手
    chk_err = max(float((per_phase[o]["turn_l"] + per_phase[o]["turn_s"]
                         - per_phase[o]["turn_all"]).abs().max()) for o in range(3))
    print(f"  换手拆分校验：多头+空头 vs 引擎总换手，最大偏差 {chk_err:.4f}（MOC 腿只付佣金，"
          f"所以允许小偏差）")

    def rate_of(sym):
        lev = abs(NOMINAL[sym])
        return (PER_SHARE / P[sym].listed.reindex(days).ffill()) / lev

    combos = [("多头 TQQQ / 空头 做空TQQQ（当前实盘，需券源）", "TQQQ", "TQQQ"),
              ("多头 TQQQ / 空头 买SQQQ(-3x)（免券源）", "TQQQ", "SQQQ"),
              ("多头 TQQQ / 空头 买QID(-2x)", "TQQQ", "QID"),
              ("多头 TQQQ / 空头 买PSQ(-1x)", "TQQQ", "PSQ"),
              ("多空都用 QQQ 本体（需买力+券源）", "QQQ", "QQQ")]
    rows3, curves = [], {}
    for lab, lsym, ssym in combos:
        rl, rs = rate_of(lsym), rate_of(ssym)
        acc = []
        for off in range(30):
            d = per_phase[off]
            acc.append(d["gross"] - d["turn_l"] * rl - d["turn_s"] * rs)
        net = pd.DataFrame(acc).T.mean(axis=1)
        curves[lab] = net
        for per, plab in ((TEST, "test23-26"), (ALL, "all19-26")):
            rows3.append(dict(方案=lab, 期间=plab,
                              多头成本bp=float(rl.loc[per[0]:per[1]].median() * 1e4),
                              空头成本bp=float(rs.loc[per[0]:per[1]].median() * 1e4),
                              **met(net.loc[per[0]:per[1]], bh.loc[per[0]:per[1]])))
    R = pd.DataFrame(rows3)
    print()
    print(R[["方案", "期间", "多头成本bp", "空头成本bp", "sharpe", "ann", "vol", "mdd",
             "beta", "alpha"]].round(3).to_string(index=False))

    # ---------- 4. 资金占用
    print("\n" + "=" * 108)
    print("4. 资金占用（目标敞口 1.95x 均值 / 3.15x p95，单位=资金倍数的名义金额）")
    print("=" * 108)
    for s in ("QQQ", "TQQQ", "SQQQ", "QID", "PSQ"):
        lev = abs(NOMINAL[s])
        print(f"  {s:5s}(|{lev:.0f}x|): 均值 {1.95/lev:.2f}x   p95 {3.15/lev:.2f}x"
              + ("   <- 实测 2500 股 QQQ 被拒" if s == "QQQ" else ""))

    R.to_csv(OUT / "inverse_etf_study.csv", index=False)
    C.to_csv(OUT / "inverse_etf_costs.csv", index=False)
    B.to_csv(OUT / "inverse_etf_betas.csv", index=False)
    pd.to_pickle(curves, OUT / "inverse_etf_curves.pkl")
    print(f"\n已写入 {OUT}/inverse_etf_*.csv")


if __name__ == "__main__":
    main()
