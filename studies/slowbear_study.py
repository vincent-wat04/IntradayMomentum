"""慢熊行情的处理 —— 按 docs/PROPOSAL_slowbear.md 冻结的规则运行。

所有规则只用 t 日收盘及以前的数据，作用于 t+1 的收益。
收益 = w·r + (1−w)·rf − 2bp·|Δw|；夏普为超额收益年化。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3_common as C
import sens_rv_weights as S

MARKETS = (("SPX", "yahoo/IDX_GSPC.parquet", 0.15), ("NDX", "yahoo/IDX_NDX.parquet", 0.20),
           ("RUT", "yahoo/IDX_RUT.parquet", 0.20), ("NYA", "yahoo/IDX_NYA.parquet", 0.15))
EPIS = {"互联网泡沫": ("1999-06-01", "2000-12-31", "2003-06-30"),
        "金融危机": ("2007-01-01", "2008-03-31", "2009-12-31"),
        "2022熊市": ("2021-06-01", "2022-03-31", "2023-03-31")}
COST = 2e-4
START = "1992-01-01"
NAMES = ["H0 浅蓝", "H1 深蓝hi=1", "H2 趋势200", "H3 动量12-1", "H4 趋势门控杠杆", "H5 回撤刹车",
         "H6 绝对波动目标", "H7 曲线倒挂", "H8 H2·H3·H5取小", "B&H"]


def rf_series(index):
    irx = C.load_close("yahoo/IDX_IRX.parquet").reindex(index).ffill()
    return irx / 100.0 / 252.0


def dd_brake(px, enter=-0.10, leave=-0.05, off=0.5):
    dd = (px / px.cummax() - 1).to_numpy()
    st = np.zeros(len(dd), bool)
    s = False
    for i, d in enumerate(dd):
        if not s and d <= enter:
            s = True
        elif s and d > leave:
            s = False
        st[i] = s
    return pd.Series(st, index=px.index)


def build_weights(px, sig_abs):
    r = px.pct_change()
    p = S.pct_w(r.rolling(21).std(), 504)
    B = S.w_linear(p, 0.5, 1.5)
    sma = px.rolling(200).mean()
    below = (px < sma).where(sma.notna(), False)
    mom = (px.shift(21) / px.shift(252) - 1)
    momneg = (mom < 0).where(mom.notna(), False)
    brake = dd_brake(px)
    rv = r.rolling(21).std() * np.sqrt(252)
    W = {"H0 浅蓝": B, "H1 深蓝hi=1": S.w_linear(p, 0.5, 1.0)}
    W["H2 趋势200"] = B.where(~below, 0.5)
    W["H3 动量12-1"] = B.where(~momneg, 0.5)
    W["H4 趋势门控杠杆"] = B.where(~below, B.clip(upper=1.0))
    W["H5 回撤刹车"] = B.where(~brake, 0.5)
    W["H6 绝对波动目标"] = (sig_abs / rv).clip(0.5, 1.5).fillna(1.0)
    tnx, irx = C.load_close("yahoo/IDX_TNX.parquet"), C.load_close("yahoo/IDX_IRX.parquet")
    spr = (tnx - irx).reindex(px.index).ffill().rolling(21).mean()
    inv = (spr < 0).where(spr.notna(), False)
    W["H7 曲线倒挂"] = B.where(~inv, B.clip(upper=1.0))
    W["H8 H2·H3·H5取小"] = pd.concat([W["H2 趋势200"], W["H3 动量12-1"], W["H5 回撤刹车"]], axis=1).min(axis=1)
    W["B&H"] = pd.Series(1.0, index=px.index)
    return W


def strat_ret(w, rr, rfn):
    turn = w.diff().abs().fillna(0.0)
    tot = w * rr + (1 - w) * rfn - COST * turn
    return tot, tot - rfn, turn


def ann_sr(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252))


def mdd_tot(x):
    eq = (1 + x.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def cagr(x):
    x = x.dropna()
    return float((1 + x).prod() ** (252 / len(x)) - 1)


def boot_dsr(a, b, nb=4000, L=10, seed=0):
    """成对循环块自助：SR(a)−SR(b) 的 95% CI。a、b 为超额收益序列。"""
    d = pd.concat([a, b], axis=1).dropna().to_numpy()
    n = len(d)
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(n / L))
    out = np.empty(nb)
    for k in range(nb):
        st = rng.integers(0, n, nblk)
        idx = (st[:, None] + np.arange(L)[None, :]).ravel()[:n] % n
        s = d[idx]
        out[k] = s[:, 0].mean() / s[:, 0].std() - s[:, 1].mean() / s[:, 1].std()
    out *= np.sqrt(252)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())


def episode_rows(px, tot_by):
    rows = {}
    for en, (a, b, c) in EPIS.items():
        seg = px.loc[a:b]; pk = seg.idxmax(); tr = px.loc[pk:c].idxmin()
        i0, i1 = px.index.get_loc(pk), px.index.get_loc(tr)
        i2 = min(i1 + 252, len(px) - 1)
        for k, tot in tot_by.items():
            dn = float((1 + tot.iloc[i0:i1].fillna(0)).prod() - 1)       # t=峰..谷前一日 作用于次日
            up = float((1 + tot.iloc[i1:i2].fillna(0)).prod() - 1)
            rows[(en, k)] = (dn, up, str(pk.date()), str(tr.date()))
    return rows


def run_market(nm, rel, sig):
    px = C.load_close(rel).dropna()
    rr = px.pct_change().shift(-1)
    rfn = rf_series(px.index)
    W = build_weights(px, sig)
    tot, ex, turn = {}, {}, {}
    for k, w in W.items():
        tot[k], ex[k], turn[k] = strat_ret(w, rr, rfn)
    return px, rr, rfn, W, tot, ex, turn


def main():
    summary = []
    for nm, rel, sig in MARKETS:
        px, rr, rfn, W, tot, ex, turn = run_market(nm, rel, sig)
        idx = rr.loc[START:].dropna().index
        print("=" * 124)
        print(f"{nm} · {idx[0].date()} → {idx[-1].date()} · 夏普为超额收益年化；对照 = 同平均权重静态组合")
        print("=" * 124)
        print(f"{'方案':18s}{'CAGR':>8}{'夏普':>7}{'MDD':>8}{'Calmar':>8}{'均权重':>8}{'换手/年':>8}"
              f"{'ΔSR(对照)':>11}{'95%CI':>17}{'P(Δ≤0)':>8}{'夏普92-09':>10}{'夏普10-26':>10}")
        for k in NAMES:
            t, e = tot[k].loc[idx], ex[k].loc[idx]
            mw = float(W[k].loc[idx].mean())
            ctrl = mw * rr + (1 - mw) * rfn - rfn
            lo, hi, pz = boot_dsr(e, ctrl.loc[idx])
            dsr = ann_sr(e) - ann_sr(ctrl.loc[idx])
            s1, s2 = ann_sr(e.loc[:"2009-12-31"]), ann_sr(e.loc["2010-01-01":])
            c, m = cagr(t), mdd_tot(t)
            print(f"{k:18s}{c:>8.1%}{ann_sr(e):>7.2f}{m:>8.1%}{c / abs(m):>8.2f}{mw:>8.2f}"
                  f"{turn[k].loc[idx].sum() / (len(idx) / 252):>8.1f}{dsr:>+11.3f}{f'[{lo:+.2f},{hi:+.2f}]':>17}{pz:>8.2f}"
                  f"{s1:>10.2f}{s2:>10.2f}")
            summary.append(dict(mkt=nm, rule=k, cagr=c, sharpe=ann_sr(e), mdd=m, calmar=c / abs(m), meanw=mw,
                                dsr=dsr, ci_lo=lo, ci_hi=hi, sr_pre2010=s1, sr_post2010=s2))
        print(f"\n  慢熊分段：{'区间':10s}{'':26s}" + "".join(f"{k[:8]:>10}" for k in NAMES))
        er = episode_rows(px, {k: tot[k] for k in NAMES})
        for en in EPIS:
            pk, trd = er[(en, "B&H")][2], er[(en, "B&H")][3]
            print(f"   {en:8s}峰{pk}→谷{trd}  峰→谷  " + "".join(f"{er[(en, k)][0]:>10.1%}" for k in NAMES))
            print(f"   {'':8s}{'':23s}谷后252日" + "".join(f"{er[(en, k)][1]:>+10.1%}" for k in NAMES))
            for k in NAMES:
                summary.append(dict(mkt=nm, rule=k, episode=en, down=er[(en, k)][0], rebound=er[(en, k)][1]))
        print()
    pd.DataFrame(summary).to_csv(Path(__file__).resolve().parent / "outputs" / "full" / "intraday" / "slowbear_study.csv", index=False)


if __name__ == "__main__":
    main()
