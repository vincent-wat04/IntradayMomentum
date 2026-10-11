"""慢熊研究的事后稳健性检查（**全部标记为事后探索**，不能改动 PROPOSAL 第 3 节的判定）。

R1  H2/H3 的回看长度是否刀口：SMA 100/150/200/250，动量 126/189/252
R2  关闭态权重 e_off = 0 / 0.5 / 1.0（1.0 = 不动，作为 H0 参照）
R3  去掉互联网泡沫这一段（只看 2003-2026）：保护还剩什么、代价是多少
R4  关闭态改持 IEF（2002-07 起）而不是现金：GFC 与 2022
R5  落到实盘形态：QQQ 腿 + 1.0×Sleeve（2019-2026，含 COVID 与 2022），沿用实盘口径（不计现金收益）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3_common as C
import sens_rv_weights as S
import slowbear_study as Z

OUT = Path("outputs/full/intraday")


def base_B(px):
    p = S.pct_w(px.pct_change().rolling(21).std(), 504)
    return S.w_linear(p, 0.5, 1.5)


def rule_w(px, B, kind, n, off=0.5):
    if kind == "sma":
        sma = px.rolling(n).mean()
        neg = (px < sma).where(sma.notna(), False)
    else:
        m = px.shift(21) / px.shift(n) - 1
        neg = (m < 0).where(m.notna(), False)
    return B.where(~neg, off), neg


def stats(px, w, a=None, b=None):
    rr = px.pct_change().shift(-1)
    rfn = Z.rf_series(px.index)
    tot, ex, _ = Z.strat_ret(w, rr, rfn)
    a = a or Z.START
    t, e = tot.loc[a:b].dropna(), ex.loc[a:b].dropna()
    return Z.cagr(t), Z.ann_sr(e), Z.mdd_tot(t)


def main():
    P = {nm: C.load_close(rel).dropna() for nm, rel, _ in Z.MARKETS if nm in ("SPX", "NDX")}
    print("=" * 110)
    print("R1  回看长度（事后）：SPX / NDX 的 全样本夏普 · MDD · 互联网泡沫峰→谷")
    print("=" * 110)
    print(f"{'规则':14s}" + "".join(f"{nm + ' 夏普':>10}{nm + ' MDD':>10}{nm + ' 泡沫':>10}" for nm in P))
    ref = {nm: (stats(px, base_B(px)), ) for nm, px in P.items()}
    row = f"{'H0 浅蓝':14s}"
    for nm, px in P.items():
        c, s, m = stats(px, base_B(px))
        er = Z.episode_rows(px, {"x": Z.strat_ret(base_B(px), px.pct_change().shift(-1), Z.rf_series(px.index))[0]})
        row += f"{s:>10.2f}{m:>10.1%}{er[('互联网泡沫', 'x')][0]:>10.1%}"
    print(row)
    for kind, ns in (("sma", (100, 150, 200, 250)), ("mom", (126, 189, 252))):
        for n in ns:
            row = f"{kind.upper() + str(n):14s}"
            for nm, px in P.items():
                w, _ = rule_w(px, base_B(px), kind, n)
                c, s, m = stats(px, w)
                er = Z.episode_rows(px, {"x": Z.strat_ret(w, px.pct_change().shift(-1), Z.rf_series(px.index))[0]})
                row += f"{s:>10.2f}{m:>10.1%}{er[('互联网泡沫', 'x')][0]:>10.1%}"
            print(row)

    print("\n" + "=" * 110)
    print("R2  关闭态权重 e_off（SMA200 与 动量12-1）：SPX / NDX 全样本夏普 · MDD")
    print("=" * 110)
    print(f"{'规则':18s}" + "".join(f"{nm + ' CAGR':>11}{nm + ' 夏普':>10}{nm + ' MDD':>10}" for nm in P))
    for kind, n in (("sma", 200), ("mom", 252)):
        for off in (0.0, 0.25, 0.5, 1.0):
            row = f"{kind.upper()}{n} off={off:<5}"
            for nm, px in P.items():
                w, _ = rule_w(px, base_B(px), kind, n, off)
                c, s, m = stats(px, w)
                row += f"{c:>11.1%}{s:>10.2f}{m:>10.1%}"
            print(row)

    print("\n" + "=" * 110)
    print("R3  去掉互联网泡沫（样本 2003-06 → 2026-10）：保护还剩多少、代价多少")
    print("=" * 110)
    print(f"{'规则':14s}" + "".join(f"{nm + ' CAGR':>11}{nm + ' 夏普':>10}{nm + ' MDD':>10}" for nm in P))
    for lab, kind, n in (("H0 浅蓝", None, 0), ("H2 SMA200", "sma", 200), ("H3 动量12-1", "mom", 252)):
        row = f"{lab:14s}"
        for nm, px in P.items():
            B = base_B(px)
            w = B if kind is None else rule_w(px, B, kind, n)[0]
            c, s, m = stats(px, w, "2003-06-01")
            row += f"{c:>11.1%}{s:>10.2f}{m:>10.1%}"
        print(row)
    row = f"{'B&H':14s}"
    for nm, px in P.items():
        c, s, m = stats(px, pd.Series(1.0, index=px.index), "2003-06-01")
        row += f"{c:>11.1%}{s:>10.2f}{m:>10.1%}"
    print(row)

    print("\n" + "=" * 110)
    print("R4  关闭态持 IEF（而非现金）：SPX，GFC 与 2022 的峰→谷（H2 SMA200，off=0.5，另 0.5 份额放 IEF/现金）")
    print("=" * 110)
    ief = C.load_close("yahoo/IEF.parquet").dropna().pct_change()
    px = P["SPX"]; rr = px.pct_change().shift(-1); rfn = Z.rf_series(px.index)
    B = base_B(px); w, neg = rule_w(px, B, "sma", 200)
    ief_n = ief.shift(-1).reindex(px.index).fillna(0.0)
    for lab, bond in (("现金", None), ("IEF", ief_n)):
        idle = (1 - w)
        alt = rfn if bond is None else bond
        tot = w * rr + idle * alt - Z.COST * w.diff().abs().fillna(0)
        er = Z.episode_rows(px, {"x": tot})
        print(f"  {lab:4s} " + "  ".join(f"{e}: 峰→谷 {er[(e, 'x')][0]:+.1%}, 谷后252日 {er[(e, 'x')][1]:+.1%}"
                                          for e in ("金融危机", "2022熊市")))
    print("  （2022 年股债同跌：IEF 当年约 −15%，所以'避险到债券'在通胀型熊市不成立。）")

    print("\n" + "=" * 110)
    print("R5  实盘形态：QQQ 腿(各规则) + 1.0×Sleeve，2019-2026（实盘口径：腿收益 = w·r − 2bp·换手，不计现金收益）")
    print("=" * 110)
    cur = pd.read_pickle(OUT / "strategy_comparison_curves.pkl")
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna()
    s = cur["多空 band+尾0.5 无周一"].reindex(bh.index).fillna(0.0)
    q = C.load_close("yahoo/QQQ.parquet").dropna()
    B = base_B(q)
    cand = {"B&H(1:1)": pd.Series(1.0, index=q.index), "H0 浅蓝": B, "H1 深蓝hi=1": S.w_linear(S.pct_w(q.pct_change().rolling(21).std(), 504), .5, 1.0),
            "H2 SMA200": rule_w(q, B, "sma", 200)[0], "H3 动量12-1": rule_w(q, B, "mom", 252)[0],
            "H2+H1(SMA200且hi=1)": rule_w(q, S.w_linear(S.pct_w(q.pct_change().rolling(21).std(), 504), .5, 1.0), "sma", 200)[0]}
    print(f"{'QQQ腿规则':22s}{'Sharpe':>8}{'CAGR':>8}{'MDD':>8}{'Calmar':>8}{'均权重':>8}{'换手/年':>8}{'COVID 2-3月':>13}{'2022全年':>10}")
    for k, wq in cand.items():
        w = wq.shift(1).reindex(bh.index).fillna(1.0)
        leg = w * bh - Z.COST * w.diff().abs().fillna(0)
        x = leg + s
        eq = (1 + x).cumprod()
        c = (eq.iloc[-1]) ** (252 / len(x)) - 1; m = float((eq / eq.cummax() - 1).min())
        covid = float((1 + x.loc["2020-02-19":"2020-03-23"]).prod() - 1)
        y22 = float((1 + x.loc["2022-01-01":"2022-12-31"]).prod() - 1)
        print(f"{k:22s}{x.mean() / x.std() * np.sqrt(252):>8.3f}{c:>8.1%}{m:>8.1%}{c / abs(m):>8.2f}{w.mean():>8.2f}"
              f"{w.diff().abs().sum() / (len(w) / 252):>8.1f}{covid:>13.1%}{y22:>10.1%}")


if __name__ == "__main__":
    main()
