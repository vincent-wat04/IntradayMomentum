"""熊市里 RV 降风险到底在做什么：红圆 lo=0 vs 浅蓝 lo=0.5。

问题：QQQ 腿 lo=0.5 永远至少持有 0.5 倍多头；红圆 lo=0 在波动最高时降到零。
直觉上 lo=0 应该更能躲熊市。直觉对不对？

"熊市"的定义必须是**实时可知**的，否则就是用后见之明挑样本：
    回撤 dd_t = P_t / max(P_0..P_t) − 1 ≤ −20%     （只用到 t 及之前的价格）
另外用公认的熊市区间（按价格自动找峰/谷，不手工挑日期）看逐段表现。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3_common as C
import sens_rv_weights as S

SPECS = {"无 RV(满仓)": None, "红圆 lo=0,hi=2": (0.0, 2.0),
         "浅蓝 lo=.5,hi=1.5": (0.5, 1.5), "深蓝 lo=.5,hi=1": (0.5, 1.0)}
MARKETS = (("SPX", "yahoo/IDX_GSPC.parquet"), ("NDX", "yahoo/IDX_NDX.parquet"))
EPIS = {  # 峰窗口起, 峰窗口止, 谷窗口止 —— 峰/谷由价格自动确定
    "互联网泡沫 2000-02": ("1999-06-01", "2000-12-31", "2003-06-30"),
    "金融危机 2007-09": ("2007-01-01", "2008-03-31", "2009-12-31"),
    "新冠崩盘 2020": ("2020-01-01", "2020-03-01", "2020-06-30"),
    "2022 熊市": ("2021-06-01", "2022-03-31", "2023-03-31"),
}
COST = 2e-4


def weights(p):
    return {k: (pd.Series(1.0, index=p.index) if v is None else S.w_linear(p, *v)) for k, v in SPECS.items()}


def main():
    for nm, rel in MARKETS:
        m = S.Mkt(nm, rel)
        px = C.load_close(rel).dropna()
        dd = px / px.cummax() - 1
        p = m.pct(21, 504)
        W = weights(p)
        rr = m.rr
        bear = (dd <= -0.20).reindex(rr.index).fillna(False)
        D = pd.DataFrame({"rr": rr, "p": p, "bear": bear}).dropna(subset=["rr"]).loc["1992":]
        for k, w in W.items():
            D["w_" + k] = w.reindex(D.index)
        print("=" * 112)
        print(f"{nm} · 1992-2026 · 熊市日 = 回撤 ≥20% 时（实时可知）：{int(D.bear.sum())} 天，占 {D.bear.mean():.1%}")
        print("=" * 112)

        print("\n[1] 熊市日里 QQQ 腿式权重到底是多少？")
        print(f"  {'设定':20s}{'熊市日 均权重':>13}{'非熊市 均权重':>14}{'熊市日 w>1 占比':>16}{'熊市日 w=0 占比':>16}")
        for k in SPECS:
            w = D["w_" + k]
            b, nb = w[D.bear], w[~D.bear]
            print(f"  {k:20s}{b.mean():>13.2f}{nb.mean():>14.2f}{(b > 1.0 + 1e-9).mean():>16.0%}{(b < 1e-9).mean():>16.0%}")

        print("\n[2] 熊市日里，按 RV 百分位三等分，次日收益（单位敞口，即指数本身）")
        print(f"  {'分组':16s}{'天数':>7}{'次日均值bp':>12}{'年化Sharpe':>12}   | 非熊市同分组：{'天数':>6}{'均值bp':>9}{'Sharpe':>8}")
        for lab, lo, hi in (("RV 低 (p<.33)", 0, .33), ("RV 中", .33, .67), ("RV 高 (p>.67)", .67, 1.01)):
            sel = (D.p >= lo) & (D.p < hi)
            a, b = D.rr[sel & D.bear], D.rr[sel & ~D.bear]
            f = lambda x: (x.mean() * 1e4, x.mean() / x.std() * np.sqrt(252)) if len(x) > 30 else (np.nan, np.nan)
            (am, asr), (bm, bsr) = f(a), f(b)
            print(f"  {lab:16s}{len(a):>7}{am:>12.1f}{asr:>12.2f}   | {'':14s}{len(b):>6}{bm:>9.1f}{bsr:>8.2f}")

        print("\n[3] 逐段熊市：峰 -> 谷 的收益，以及谷之后 126 个交易日的反弹期收益")
        hdr = f"  {'区间':20s}{'峰':>11}{'谷':>11}{'指数跌幅':>9}" + "".join(f"{k[:9]:>13}" for k in SPECS)
        print(hdr)
        for en, (a, b, c) in EPIS.items():
            seg = px.loc[a:b]; pk = seg.idxmax()
            tr = px.loc[pk:c].idxmin()
            i0, i1 = px.index.get_loc(pk), px.index.get_loc(tr)
            idx = px.index[i0:i1]                       # t = 峰..谷前一日，作用于 t+1 的收益
            row = f"  {en:20s}{str(pk.date()):>11}{str(tr.date()):>11}{px.loc[tr]/px.loc[pk]-1:>9.1%}"
            row2 = f"  {'  └ 平均权重 / 其后126日':20s}{'':>11}{'':>11}{'':>9}"
            for k in SPECS:
                w = W[k].reindex(px.index)
                x = (w * m.rr - COST * w.diff().abs().fillna(0)).reindex(idx)
                ret = float((1 + x.fillna(0)).prod() - 1)
                idx2 = px.index[i1:min(i1 + 126, len(px) - 1)]
                x2 = (w * m.rr - COST * w.diff().abs().fillna(0)).reindex(idx2)
                reb = float((1 + x2.fillna(0)).prod() - 1)
                row += f"{ret:>13.1%}"
                row2 += f"{w.reindex(idx).mean():>6.2f}/{reb:>+6.0%}"
            print(row); print(row2)

    # 机制演示：互联网泡沫期间，百分位如何随时间"适应"高波动
    print("\n" + "=" * 112)
    print("[4] 机制：SPX 2000-03 ~ 2003-03，RV 百分位是相对**自己过去 504 天**的位置 —— 持续高波动会让它自己'降'下来")
    print("=" * 112)
    px = C.load_close("yahoo/IDX_GSPC.parquet").dropna()
    m = S.Mkt("SPX", "yahoo/IDX_GSPC.parquet")
    p = m.pct(21, 504); dd = px / px.cummax() - 1
    rv = px.pct_change().rolling(21).std() * np.sqrt(252) * 100
    W = weights(p)
    print(f"  {'日期':>11}{'回撤':>8}{'RV21年化%':>11}{'RV百分位':>10}{'w 红圆':>9}{'w 浅蓝':>9}")
    for d in px.loc["2000-03-01":"2003-03-31"].index[::63]:
        print(f"  {str(d.date()):>11}{dd.loc[d]:>8.0%}{rv.loc[d]:>11.1f}{p.loc[d]:>10.2f}"
              f"{W['红圆 lo=0,hi=2'].loc[d]:>9.2f}{W['浅蓝 lo=.5,hi=1.5'].loc[d]:>9.2f}")


if __name__ == "__main__":
    main()
