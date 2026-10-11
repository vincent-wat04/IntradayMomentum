"""日内横截面策略：隔夜跳空反转、首半小时动量，全部**不持隔夜仓**（收盘 MOC 平仓）。

动机（为什么单标的测完还要测横截面）：
  Lou, Polk & Skouras (2018, JFE, "A tug of war: Overnight versus intraday expected returns")
  的结论是**横截面**的：把每只股票的收益拆成隔夜段和日内段后，动量策略的利润几乎全部来自
  隔夜段，而很多其他策略（尤其是反转类）的利润几乎全部来自日内段。这是对**股票之间相对
  排序**的论断，不是对指数 ETF 时序择时的论断。所以在 SPY/QQQ 上测不出来，不代表不存在。
  相关证据：Berkman, Koch, Tuttle & Zhang (2012, JFQA) 个人投资者关注推高开盘价、日内回落；
  Branch & Ma (2012) 隔夜与日内收益的负相关；Heston, Korajczyk & Sadka (2010, JF) 日内周期性。

数据：data/min5_us/*.csv，77 只美股 5 分钟（moomoo，**bar 结束时标**，前复权）
     所以标签 09:35 的 bar 覆盖 09:30-09:35，标签 16:00 的 bar 覆盖 15:55-16:00。

时序约定（严格无前视）：
  隔夜跳空 gap = 09:30 开盘价 / 前收 − 1，在 09:30 就已知 -> 09:35 成交（标签 09:40 的 bar 开盘）
  首半小时收益 = 10:00 价 / 09:30 开盘 − 1，10:00 已知 -> 10:00 成交（标签 10:05 的 bar 开盘）
  末半小时持仓 = 15:30 成交（标签 15:35 的 bar 开盘）-> 收盘平仓
  所有策略都在当日收盘平仓，隔夜敞口严格为 0。

成本：半价差 0.5¢ + 每股佣金 $0.0035 -> 相对成本 = 0.0085/价格（逐只、逐日），再加滑点。
     个股比 QQQ 难做，滑点基准取 1.0bp（QQQ 是 0.32bp）；另给 3bp/5bp 的敏感性。
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
SRC = Path("data/min5_us")
PER_SHARE = 0.005 + 0.0035
SLIP = 1.0          # bp，个股滑点基准
HORIZ = ["10:00", "10:30", "11:00", "12:00", "13:00", "14:00", "15:00", "15:30"]


def daily_features(path):
    """每只股票 -> 每日一行：各个成交时点的价格 + 信号所需的价格。"""
    d = pd.read_csv(path, usecols=["time_key", "open", "high", "low", "close", "volume", "turnover"])
    ts = pd.to_datetime(d.time_key)
    d["day"] = ts.dt.strftime("%Y-%m-%d")
    d["t"] = ts.dt.strftime("%H:%M")
    d = d[(d.t > "09:30") & (d.t <= "16:00")]
    n = d.groupby("day").t.size()
    d = d[d.day.isin(n[n >= 70].index)]
    g = d.groupby("day")
    out = pd.DataFrame({
        "o_open": g.open.first(),                     # 09:30 开盘
        "dayclose": g.close.last(),                   # 16:00 收盘
        "dollar_vol": g.turnover.sum(),
    })
    piv_o = d.pivot_table(index="day", columns="t", values="open", aggfunc="first")
    piv_c = d.pivot_table(index="day", columns="t", values="close", aggfunc="first")
    want = [("o_0935", "09:40", piv_o), ("c_1000", "10:00", piv_c),
            ("o_1000", "10:05", piv_o), ("o_1530", "15:35", piv_o), ("c_1530", "15:30", piv_c)]
    # 持仓时长扫描用：信号 = 开盘->T 的收益（T 时刻的 bar 收盘价），成交价 = T+5min 的 bar 开盘价
    for T in HORIZ:
        te = (pd.Timestamp(f"2000-01-01 {T}") + pd.Timedelta("5min")).strftime("%H:%M")
        want += [(f"c_{T.replace(':','')}", T, piv_c), (f"o_{T.replace(':','')}", te, piv_o)]
    for lab, col, src in want:
        out[lab] = src[col] if col in src.columns else np.nan
    out["prev_close"] = out.dayclose.shift(1)
    return out


def build_panel():
    cache = OUT / "xs_intraday_panel.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    frames = {}
    for i, f in enumerate(sorted(SRC.glob("*.csv"))):
        frames[f.stem] = daily_features(f)
        if (i + 1) % 20 == 0:
            print(f"  loaded {i+1} symbols")
    panel = {k: pd.concat({s: df[k] for s, df in frames.items()}, axis=1)
             for k in next(iter(frames.values())).columns}
    panel = {k: v.sort_index() for k, v in panel.items()}
    pd.to_pickle(panel, cache)
    return panel


def ls_book(signal, sign, frac=0.2, min_n=20):
    """横截面 rank -> 多空权重。sign=+1 顺着信号做多高分，−1 做多低分。
    权重 = 去均值的横截面 rank，再按绝对值之和归一到总杠杆 1（多 0.5 空 0.5）。"""
    S = signal * sign
    r = S.rank(axis=1, pct=True)
    ok = S.notna().sum(axis=1) >= min_n
    w = r.sub(r.mean(axis=1), axis=0)
    # 只保留两端 frac 分位，中间置 0（分组回测的连续版本）
    lo = r.quantile(frac, axis=1); hi = r.quantile(1 - frac, axis=1)
    w = w.where(r.le(lo, axis=0) | r.ge(hi, axis=0), 0.0)
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(gross, axis=0)
    return w.where(ok, np.nan)


def run_strategy(panel, name, sig, sign, entry_px, slip=SLIP, frac=0.2):
    """entry_px: 入场成交价 DataFrame；出场一律当日收盘价。"""
    ex = panel["dayclose"]
    ret = ex / entry_px - 1                                   # 入场成交价 -> 收盘，可实现
    cost_bp = (PER_SHARE / entry_px) * 1e4 + slip             # 逐只逐日，入场 1 边
    cost_bp_exit = (PER_SHARE / ex) * 1e4 + slip              # 出场 1 边（MOC）
    w = ls_book(sig, sign, frac)
    valid = ret.notna() & entry_px.notna() & w.notna()
    w = w.where(valid, 0.0)
    gross = (w * ret.where(valid, 0.0)).sum(axis=1)
    cst = (w.abs() * (cost_bp + cost_bp_exit).where(valid, 0.0)).sum(axis=1) * 1e-4
    net = (gross - cst).replace(0, np.nan)
    ic = ret.corrwith((sig * sign).rank(axis=1), axis=1, method="spearman")
    return dict(name=name, ret=net, gross=gross, cost=cst, ic=ic,
                n=valid.sum(axis=1), mean_cost_bp=float(cost_bp.stack().median()))


def metrics(x):
    x = x.dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(sharpe=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    return dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)), ann=float(x.mean() * 252),
                vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()), hit=float((x > 0).mean()),
                t_stat=float(x.mean() / x.std() * np.sqrt(len(x))), n_days=int(len(x)))


def main():
    print("载入 77 只 5 分钟数据 -> 日频特征面板 ...")
    P = build_panel()
    P = {k: v.set_axis(pd.to_datetime(v.index)) for k, v in P.items()}
    days = P["dayclose"].index
    print(f"  {P['dayclose'].shape[1]} 只，{len(days)} 天，{days[0]} .. {days[-1]}")

    # 流动性过滤：当日前 60 名美元成交额（全部是大盘股，这步基本不筛掉什么，但保持一致）
    dv = P["dollar_vol"]
    liq = dv.rank(axis=1, ascending=False) <= 60

    gap = P["o_open"] / P["prev_close"] - 1                 # 隔夜跳空，09:30 已知
    r_fh = P["c_1000"] / P["o_open"] - 1                   # 首半小时，10:00 已知
    r_day = P["c_1530"] / P["o_open"] - 1                  # 开盘到 15:30，15:30 已知
    for X in (gap, r_fh, r_day):
        X.mask(~liq, inplace=True)

    specs = [
        # name,                         signal, sign, 入场价,            说明
        ("XS_ON_REV  隔夜跳空反转",      gap,   -1, P["o_0935"]),
        ("XS_ON_CONT 隔夜跳空延续",      gap,   +1, P["o_0935"]),
        ("XS_FH_REV  首半小时反转",      r_fh,  -1, P["o_1000"]),
        ("XS_FH_CONT 首半小时延续",      r_fh,  +1, P["o_1000"]),
        ("XS_GAO     首半小时->末半小时", r_fh,  +1, P["o_1530"]),
        ("XS_L30_REV 全日反转->末半小时", r_day, -1, P["o_1530"]),
        ("XS_L30_CONT全日动量->末半小时", r_day, +1, P["o_1530"]),
    ]
    res, rows = {}, []
    for slip in (1.0, 3.0, 5.0):
        for nm, s, sg, px in specs:
            o = run_strategy(P, nm, s, sg, px, slip=slip)
            for per, lab in [(slice(None), "all 2019-2026"), (slice("2023", None), "test 2023-2026")]:
                rows.append(dict(strat=nm, slip_bp=slip, period=lab,
                                 ic=float(o["ic"].loc[per].mean()),
                                 icir=float(o["ic"].loc[per].mean() / o["ic"].loc[per].std()),
                                 gross_sr=metrics(o["gross"].loc[per]).get("sharpe"),
                                 cost_bp_per_leg=o["mean_cost_bp"],
                                 **metrics(o["ret"].loc[per])))
            if slip == 1.0:
                res[nm] = o

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "xs_intraday.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)
    c = ["strat", "slip_bp", "period", "ic", "icir", "gross_sr", "sharpe", "ann", "vol", "mdd",
         "t_stat", "cost_bp_per_leg", "n_days"]
    print("\n=== 日内横截面策略（20%/20% 多空，收盘平仓，隔夜敞口=0）===")
    print(R[c].round(3).to_string(index=False))

    print("\n=== 毛收益 vs 成本（滑点 1bp，全样本年化）===")
    for nm, o in res.items():
        print(f"  {nm}: 毛 {o['gross'].mean()*252:+.2%}/年, 成本 {o['cost'].mean()*252:.2%}/年, "
              f"净 {o['ret'].mean()*252:+.2%}/年, 日均双边成本 {o['mean_cost_bp']*2:.2f}bp")

    # ---------- 持仓时长扫描：同一个"日内反转"信号，入场时点越早 -> 单次往返能吃到的毛收益越多
    print("\n=== 持仓时长扫描：信号 = 开盘->T 的横截面收益排序（反转方向），T 入场、收盘平仓 ===")
    print("    一次往返的成本是固定的（约 3.3bp），所以问题是：持仓多久才攒够超过成本的毛收益？")
    hrows = []
    for T in HORIZ:
        key = T.replace(":", "")
        c_T, o_T = P[f"c_{key}"].mask(~liq), P[f"o_{key}"]
        sig = c_T / P["o_open"] - 1
        o = run_strategy(P, f"T={T}", sig, -1, o_T, slip=1.0)
        mins = (pd.Timestamp("2000-01-01 16:00") - pd.Timestamp(f"2000-01-01 {T}")).seconds / 60
        g = o["gross"].dropna()
        hrows.append(dict(entry=T, hold_min=mins, ic=float(o["ic"].mean()),
                          icir=float(o["ic"].mean() / o["ic"].std()),
                          gross_bp_per_day=float(g.mean() * 1e4),
                          roundtrip_cost_bp=o["mean_cost_bp"] * 2,
                          gross_sr=metrics(g)["sharpe"], **metrics(o["ret"])))
        res[f"HOLD T={T}"] = o
    H = pd.DataFrame(hrows)
    H["net_bp_per_day"] = H.gross_bp_per_day - H.roundtrip_cost_bp
    # 反事实上限：滑点严格为 0（不可能做到），只付 0.5¢ 半价差 + $0.0035 佣金
    H["cost_zero_slip_bp"] = H.roundtrip_cost_bp - 2 * 1.0
    H["net_zero_slip_bp"] = H.gross_bp_per_day - H.cost_zero_slip_bp
    H["net_zero_slip_ann"] = H.net_zero_slip_bp * 252 * 1e-4
    H.to_csv(OUT / "xs_intraday_horizon.csv", index=False)
    print(H[["entry", "hold_min", "ic", "icir", "gross_sr", "gross_bp_per_day",
             "roundtrip_cost_bp", "net_bp_per_day", "sharpe",
             "cost_zero_slip_bp", "net_zero_slip_bp", "net_zero_slip_ann"]].round(3).to_string(index=False))
    print("    -> 即使滑点严格为 0（物理上做不到），最好的一档也只有 "
          f"{H.net_zero_slip_ann.max():.2%}/年（gross=1 的多空账本，波动约 "
          f"{H.loc[H.net_zero_slip_bp.idxmax(),'vol']:.1%}）。")

    print("\n=== 两两相关（滑点 1bp, test 期）===")
    cc = pd.DataFrame({nm.split()[0]: res[nm]["ret"] for nm, _, _, _ in specs}).loc["2023":].corr()
    print(cc.round(2).to_string())
    cc.to_csv(OUT / "xs_intraday_corr.csv")

    names = [nm for nm, _, _, _ in specs]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    ax = axes[0]
    for nm in names:
        s = res[nm]["gross"].dropna()
        ax.plot((1 + s).cumprod(), lw=1.1, label=f"{nm.split()[0]} gross SR {metrics(s)['sharpe']:.2f}")
    ax.set_title("gross (no cost)"); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_yscale("log")
    ax = axes[1]
    for nm in names:
        s = res[nm]["ret"].dropna()
        ax.plot((1 + s).cumprod(), lw=1.1, label=f"{nm.split()[0]} net SR {metrics(s)['sharpe']:.2f}")
    ax.axhline(1, color="k", lw=.8); ax.set_title("net of cost (slip 1bp)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[2]
    ax.plot(H.hold_min, H.gross_bp_per_day, "o-", label="gross bp/day")
    ax.plot(H.hold_min, H.roundtrip_cost_bp, "s--", label="round-trip cost bp")
    ax.plot(H.hold_min, H.net_bp_per_day, "^-", label="net bp/day")
    ax.axhline(0, color="k", lw=.8); ax.set_xlabel("holding minutes")
    ax.set_title("intraday XS reversal: edge vs cost")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_xs_intraday.png", dpi=110); plt.close()
    print("\nfigure written")
    return R, res


if __name__ == "__main__":
    main()
