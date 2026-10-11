"""对 QQQ 腿卖月度备兑看涨：合成定价回测。严格按 docs/PROPOSAL_overwrite.md。

定价：Black-Scholes，ATM 隐含波动率 σ_t = m × VXN_t/100，r = IRX，q = 0.6%，T = 剩余交易日/252。
持仓：每股 QQQ 1 股 + 卖 1 份看涨（覆盖 100%）。
损益（每股，以前一日原始价格 S_{t-1} 为分母）：
    r_cc,t = r_QQQ,t(复权总回报) − (C_t − C_{t-1})/S_{t-1} − 成本_t/S_{t-1}
其中 C_t 是当日按 VXN_t 与剩余期限盯市的看涨价；到期日 C = max(S−K, 0)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3_common as C
from stop_significance import block_boot_pair

OUT = Path("outputs/full/intraday")
ROLL = 21
DIV = 0.006
KS = (0.0, 0.02, 0.05)
MS = (0.80, 0.90, 1.00)
HEAD = (0.02, 0.90)
SPREAD, ABS, COMM = 0.03, 0.01, 0.0065


def bs_call(S, K, T, r, q, sig):
    """Black-Scholes 看涨价与 delta。标量和数组都可以；T<=0 时返回内在价值。"""
    S, K, T, r, q, sig = [np.atleast_1d(np.asarray(x, float)).copy() for x in np.broadcast_arrays(S, K, T, r, q, sig)]
    out = np.maximum(S - K, 0.0)
    dl = (S > K).astype(float)
    ok = (T > 1e-9) & (sig > 1e-9)
    if ok.any():
        sT = sig[ok] * np.sqrt(T[ok])
        d1 = (np.log(S[ok] / K[ok]) + (r[ok] - q[ok] + 0.5 * sig[ok] ** 2) * T[ok]) / sT
        d2 = d1 - sT
        out[ok] = S[ok] * np.exp(-q[ok] * T[ok]) * norm.cdf(d1) - K[ok] * np.exp(-r[ok] * T[ok]) * norm.cdf(d2)
        dl[ok] = np.exp(-q[ok] * T[ok]) * norm.cdf(d1)
    return out, dl


def sr(x):
    x = pd.Series(x).dropna(); return float(x.mean() / x.std() * np.sqrt(252))


def cagr(x):
    x = pd.Series(x).dropna(); return float((1 + x).prod() ** (252 / len(x)) - 1)


def mdd(x):
    eq = (1 + pd.Series(x).dropna()).cumprod(); return float((eq / eq.cummax() - 1).min())


def load():
    q = pd.read_parquet(C.RAW / "yahoo/QQQ.parquet")
    q.index = pd.to_datetime(q.index).tz_localize(None)
    adj = q["close"]; raw = q["close_raw"] if "close_raw" in q.columns else q["close"]
    vxn = C.load_close("cboe/VXN.parquet")
    rf = C.load_close("yahoo/IDX_IRX.parquet") / 100.0
    D = pd.DataFrame({"adj": adj, "raw": raw}).join(vxn.rename("vxn"), how="inner").join(rf.rename("rf"), how="left")
    D["rf"] = D["rf"].ffill().fillna(0.02)
    D["r"] = D.adj.pct_change()
    return D.dropna(subset=["vxn", "r"]).loc["2009-09-15":]


def overwrite(D, k, m):
    """返回 (日收益 Series, 日净 delta Series, 每月权利金%, 每月被行权%)。"""
    S = D.raw.to_numpy(); n = len(D); vx = m * D.vxn.to_numpy() / 100.0; rf = D.rf.to_numpy()
    r = D.r.to_numpy()
    ret = np.zeros(n); delta = np.ones(n); prem, itm = [], []
    i0 = 0
    K = None; te = None; C_prev = 0.0
    for t in range(n):
        if t == i0:
            # 开仓日：当日收盘才卖出期权。所以当日的**股票收益 r[t] 照常计入**（前一份期权已在上一个到期日结算），
            # 期权盯市损益为 0（还没开仓），只扣开仓成本。
            # （第一版把股票收益丢了，每个月少算一天，已修。）
            K = S[t] * (1 + k); te = t + ROLL
            c0, dl0 = bs_call(S[t], K, ROLL / 252, rf[t], DIV, vx[t])
            c0, dl0 = float(c0[0]), float(dl0[0])
            cost = c0 * SPREAD + ABS + COMM
            ret[t] = r[t] - cost / S[t - 1]
            delta[t] = 1 - dl0; C_prev = c0; prem.append(c0 * (1 - SPREAD) / S[t])
            continue
        Tleft = (te - t) / 252
        c, dl = bs_call(S[t], K, max(Tleft, 0.0), rf[t], DIV, vx[t])
        c, dl = float(c[0]), float(dl[0])
        ret[t] = r[t] - (c - C_prev) / S[t - 1]
        delta[t] = 1 - (dl if Tleft > 0 else float(S[t] > K))
        C_prev = c
        if t == te:
            itm.append(float(S[t] > K)); i0 = t + 1
            # 到期当日收盘后立刻卖新一份：下一交易日作为开仓日
    return pd.Series(ret, index=D.index), pd.Series(delta, index=D.index), np.mean(prem), (np.mean(itm) if itm else np.nan)


def metrics(x, bench, rf, dbar):
    ctrl = dbar * bench + (1 - dbar) * rf
    X = sm.add_constant(bench.to_numpy())
    mdl = sm.OLS(x.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return dict(sharpe=sr(x), cagr=cagr(x), mdd=mdd(x), beta=float(mdl.params[1]), alpha=float(mdl.params[0]) * 252,
                t=float(mdl.tvalues[0]), ctrl=ctrl, c_sharpe=sr(ctrl), c_cagr=cagr(ctrl), c_mdd=mdd(ctrl))


def main():
    D = load()
    rfd = D.rf / 252
    bench = D.r
    print("=" * 118)
    print(f"样本 {D.index[0].date()} ~ {D.index[-1].date()}（{len(D)} 天）；VXN 均值 {D.vxn.mean():.1f}；"
          f"QQQ 买入持有 Sharpe {sr(bench):.3f}、CAGR {cagr(bench):.1%}、MDD {mdd(bench):.1%}")
    print("=" * 118)
    print(f"{'k':>6}{'m':>6}{'月权利金%':>10}{'被行权%':>9}{'平均delta':>10}{'Sharpe':>8}{'CAGR':>8}{'MDD':>8}"
          f"{'β':>7}{'α年化':>8}{'α的t':>7} | {'对照SR':>7}{'对照CAGR':>9}{'对照MDD':>9} | {'ΔSR':>7}{'ΔCAGR':>8}{'ΔMDD':>7}")
    res = {}; rows = []
    for k in KS:
        for m in MS:
            x, dl, pm, it = overwrite(D, k, m)
            x = x.iloc[1:]; dbar = float(dl.iloc[1:].mean())
            M = metrics(x, bench.loc[x.index], rfd.loc[x.index], dbar)
            res[(k, m)] = (x, dl, M, dbar)
            tag = " ←headline" if (k, m) == HEAD else ""
            print(f"{k:>6.0%}{m:>6.2f}{pm*100:>10.2f}{it*100:>9.0f}{dbar:>10.2f}{M['sharpe']:>8.3f}{M['cagr']:>8.1%}{M['mdd']:>8.1%}"
                  f"{M['beta']:>7.2f}{M['alpha']:>8.2%}{M['t']:>7.2f} | {M['c_sharpe']:>7.3f}{M['c_cagr']:>9.1%}{M['c_mdd']:>9.1%} | "
                  f"{M['sharpe']-M['c_sharpe']:>+7.3f}{(M['cagr']-M['c_cagr'])*100:>+7.1f}pp{(M['mdd']-M['c_mdd'])*100:>+6.1f}{tag}")
            rows.append(dict(k=k, m=m, premium_pct=pm, itm=it, dbar=dbar, **{a: b for a, b in M.items() if a not in ("ctrl",)}))
    pd.DataFrame(rows).to_csv(OUT / "overwrite_study.csv", index=False)

    x, dl, M, dbar = res[HEAD]
    print("\n" + "=" * 118)
    print(f"headline（k={HEAD[0]:.0%}, m={HEAD[1]}）细节")
    print("=" * 118)
    ctrl = M["ctrl"]
    r, dist = block_boot_pair(x, ctrl.loc[x.index], nb=1000, L=10, seed=3)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    print(f"  ΔSharpe(备兑 − 匹配 delta 对照) = {r:+.3f}，95% 区间 [{lo:+.3f}, {hi:+.3f}]")
    print(f"  上行/下行捕获（相对满仓 QQQ，日度回归斜率）：上涨日 β={np.polyfit(bench.loc[x.index][bench.loc[x.index]>0], x[bench.loc[x.index]>0], 1)[0]:.2f}，"
          f"下跌日 β={np.polyfit(bench.loc[x.index][bench.loc[x.index]<0], x[bench.loc[x.index]<0], 1)[0]:.2f}  （备兑要'保护'应在下跌日 β 明显更低）")
    print(f"\n  {'区间':22s}{'满仓QQQ':>9}{'备兑':>9}{'对照':>9}{'备兑−对照':>11}")
    for nm, a, b in (("新冠崩盘 2020-02-19~03-23", "2020-02-19", "2020-03-23"), ("2022 全年", "2022-01-03", "2022-12-30"),
                     ("2018Q4 回调 10-01~12-24", "2018-10-01", "2018-12-24"), ("牛市 2023-01~2026-10", "2023-01-03", "2026-10-02"),
                     ("2009-10~2019-12 十年牛市", "2009-10-01", "2019-12-31")):
        sl = slice(a, b)
        f = lambda s: float((1 + s.loc[sl]).prod() - 1)
        print(f"  {nm:22s}{f(bench):>9.1%}{f(x):>9.1%}{f(ctrl):>9.1%}{f(x)-f(ctrl):>+11.1%}")

    # 组合层：QQQ 腿(静态 1×) + Sleeve，2019-2026
    print("\n" + "=" * 118)
    print("组合层（2019-2026）：QQQ 腿 + 1.0×Sleeve —— Sleeve 是长凸性，卖 call 是短凸性")
    print("=" * 118)
    cur = pd.read_pickle(OUT / "strategy_comparison_curves.pkl")
    s = cur["多空 band+尾0.5 无周一"].copy(); s.index = pd.to_datetime(s.index)
    idx = s.index.intersection(x.index)
    s = s.loc[idx]
    print(f"  {'组合':32s}{'Sharpe':>8}{'CAGR':>8}{'MDD':>8}{'Calmar':>8}")
    for nm, leg in (("QQQ 满仓 + Sleeve", bench.loc[idx]), ("备兑 QQQ(headline) + Sleeve", x.loc[idx]),
                    ("匹配 delta 对照 + Sleeve", ctrl.loc[idx])):
        c = leg + s
        print(f"  {nm:32s}{sr(c):>8.3f}{cagr(c):>8.1%}{mdd(c):>8.1%}{cagr(c)/abs(mdd(c)):>8.2f}")
    print("\n决定规则（proposal §5）：")
    hx = res[HEAD][2]; h8 = res[(HEAD[0], 0.80)][2]
    a = (hx["sharpe"] - hx["c_sharpe"] > 0.15) and abs(hx["t"]) > 2
    b = (h8["sharpe"] - h8["c_sharpe"] > 0.15) and abs(h8["t"]) > 2
    cks = [np.sign(res[(k, 0.90)][2]["sharpe"] - res[(k, 0.90)][2]["c_sharpe"]) for k in KS]
    c_ok = len(set(cks)) == 1
    print(f"  (a) headline ΔSR>0.15 且 |t|>2 : {'是' if a else '**否**'}   (ΔSR={hx['sharpe']-hx['c_sharpe']:+.3f}, t={hx['t']:.2f})")
    print(f"  (b) m=0.80 仍成立             : {'是' if b else '**否**'}   (ΔSR={h8['sharpe']-h8['c_sharpe']:+.3f}, t={h8['t']:.2f})")
    print(f"  (c) 三个行权价 ΔSR 同号        : {'是' if c_ok else '**否**'}   {cks}")


if __name__ == "__main__":
    main()
