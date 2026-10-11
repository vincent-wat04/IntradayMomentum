"""恐慌族因子能否降低组合回撤 —— 按 docs/PROPOSAL_fear_derisk.md 执行。

主指标是 MDD 而非 Sharpe；最关键的控制是**敞口匹配**
（所有变体的 w_qqq 样本均值相同），否则"恐慌高就减仓"会因为
"平均持股更少"而机械降低回撤，那不是择时能力。
"""
from __future__ import annotations

import paths as _P
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("outputs/full/intraday")
CBOE = (_P.RAW / "cboe")
DEV, TEST = ("2019", "2022"), ("2023", "2026")
WIN, MINP = 504, 252          # 因果百分位窗口
N_PLACEBO = 200
VER = "多空 band+尾0.5 无周一"


def sr(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(252))


def mdd(x):
    eq = (1 + pd.Series(x).dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def cagr(x):
    x = pd.Series(x).dropna(); eq = (1 + x).cumprod()
    return float(eq.iloc[-1] ** (252 / len(x)) - 1)


def stats(x):
    m = mdd(x)
    return dict(sharpe=sr(x), cagr=cagr(x), mdd=m,
                calmar=(cagr(x) / abs(m) if m < 0 else np.nan))


def causal_pct(x: pd.Series) -> pd.Series:
    """因果百分位：t 时刻的值在过去 WIN 个观测中的分位。只用过去，不含当期以后。"""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        lo = max(0, i - WIN + 1)
        w = v[lo:i + 1]
        w = w[np.isfinite(w)]
        if len(w) >= MINP and np.isfinite(v[i]):
            out[i] = float((w < v[i]).mean())
    return pd.Series(out, index=x.index)


def weight_from(pct: pd.Series, direction: str, target_mean: float,
                lo: float = 0.0, hi: float = 2.0) -> pd.Series:
    """百分位 -> QQQ 权重，再整体缩放到指定均值（敞口匹配）。"""
    p = pct.copy()
    raw = (1.0 - p) if direction == "derisk" else p
    raw = lo + (hi - lo) * raw
    raw = raw.fillna(target_mean)          # 预热期用常数，不引入前视
    m = raw.mean()
    return raw * (target_mean / m) if m > 0 else raw


def main():
    cur = pd.read_pickle(OUT / "strategy_comparison_curves.pkl")
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna()
    s = cur[VER].reindex(bh.index).fillna(0.0)
    idx = bh.index

    # ---- 信号（t−1）
    sig = {}
    for nm, f in (("VIX", "VIX"), ("COR3M", "COR3M"), ("VVIX", "VVIX")):
        d = pd.read_parquet(CBOE / f"{f}.parquet")
        col = "close" if "close" in d.columns else d.columns[0]
        sig[nm] = d[col].reindex(idx).ffill().shift(1)
    sig["RV21"] = bh.rolling(21).std().shift(1)        # 土办法基准
    pct = {k: causal_pct(v) for k, v in sig.items()}
    print("=== 信号覆盖率（非 NaN 百分位的天数占比）===")
    for k, v in pct.items():
        print(f"  {k:7s} {v.notna().mean():.1%}   与 QQQ 次日收益相关 "
              f"{float(pd.Series(v).corr(bh)):+.4f}")

    print("\n=== 对照基准 ===")
    rows = []
    for nm, q, sw in (("B0 静态 (1.0, 1.0)", 1.0, 1.0),
                      ("B1 静态 (0.6, 1.4)", 0.6, 1.4)):
        c = q * bh + sw * s
        st = stats(c); st["组"] = nm; st["平均w_qqq"] = q; rows.append(st)
        print(f"  {nm:22s} Sharpe {st['sharpe']:.3f}  CAGR {st['cagr']:.2%}  "
              f"MDD {st['mdd']:.2%}  Calmar {st['calmar']:.3f}")

    # ---- 动态变体：敞口匹配到 mean(w_qqq)=1.0，sleeve 固定 1.0
    print("\n=== 动态变体（敞口匹配：mean(w_qqq)=1.0，sleeve=1.0）===")
    print(f"  {'组':30s}{'Sharpe':>9}{'CAGR':>9}{'MDD':>9}{'Calmar':>9}"
          f"{'w范围':>14}")
    dyn = {}
    for k in ("VIX", "COR3M", "VVIX", "RV21"):
        for dr, drn in (("derisk", "降风险"), ("riskprem", "风险溢价")):
            w = weight_from(pct[k], dr, 1.0)
            c = w * bh + 1.0 * s
            st = stats(c)
            nm = f"{k} · {drn}"
            dyn[(k, dr)] = (c, w)
            st["组"] = nm; st["平均w_qqq"] = float(w.mean()); rows.append(st)
            print(f"  {nm:30s}{st['sharpe']:>9.3f}{st['cagr']:>9.2%}"
                  f"{st['mdd']:>9.2%}{st['calmar']:>9.3f}"
                  f"{f'{w.min():.2f}~{w.max():.2f}':>14}")

    b0 = stats(1.0 * bh + 1.0 * s)
    b1 = stats(0.6 * bh + 1.4 * s)

    # ---- 循环平移安慰剂（保留自相关、破坏与收益的对齐）
    print(f"\n=== 循环平移安慰剂（{N_PLACEBO} 次，主指标 = MDD 改善）===")
    rng = np.random.default_rng(20261007)
    for k in ("VIX", "COR3M"):
        real = dyn[(k, "derisk")][0]
        d_real = mdd(real) - b0["mdd"]        # >0 表示回撤变浅（改善）
        null = []
        p = pct[k].to_numpy()
        for _ in range(N_PLACEBO):
            sh = int(rng.integers(60, len(p) - 60))
            ps = pd.Series(np.roll(p, sh), index=idx)
            w = weight_from(ps, "derisk", 1.0)
            null.append(mdd(w * bh + s) - b0["mdd"])
        null = np.array(null)
        print(f"  {k:7s} 真实 MDD 改善 {d_real:+.4f}  零分布 均值 {null.mean():+.4f} "
              f"std {null.std():.4f} p90 {np.percentile(null,90):+.4f} "
              f"-> 分位 {float((null<d_real).mean()):.0%} "
              f"{'超过p90' if d_real>np.percentile(null,90) else '**未超p90**'}")

    # ---- dev / test 一致性
    print("\n=== dev / test 一致性（MDD 与 Sharpe，vs B0）===")
    print(f"  {'组':30s}{'dev ΔMDD':>11}{'test ΔMDD':>11}{'dev ΔSR':>10}{'test ΔSR':>10}")
    for k in ("VIX", "COR3M", "VVIX", "RV21"):
        c = dyn[(k, "derisk")][0]
        line = f"  {k+' · 降风险':30s}"
        for per in (DEV, TEST):
            a = c.loc[per[0]:per[1]]; bb = (bh + s).loc[per[0]:per[1]]
            line += f"{mdd(a)-mdd(bb):>11.4f}"
        for per in (DEV, TEST):
            a = c.loc[per[0]:per[1]]; bb = (bh + s).loc[per[0]:per[1]]
            line += f"{sr(a)-sr(bb):>10.3f}"
        print(line)

    # ---- 把动态方案叠在 B1 之上：还有没有增量？
    print("\n=== 在静态再配比 (0.6, 1.4) 之上，动态择时还有增量吗？ ===")
    print(f"  {'组':30s}{'Sharpe':>9}{'CAGR':>9}{'MDD':>9}{'Calmar':>9}")
    print(f"  {'B1 静态 (0.6, 1.4)':30s}{b1['sharpe']:>9.3f}{b1['cagr']:>9.2%}"
          f"{b1['mdd']:>9.2%}{b1['calmar']:>9.3f}")
    for k in ("VIX", "COR3M", "RV21"):
        w = weight_from(pct[k], "derisk", 0.6)
        c = w * bh + 1.4 * s
        st = stats(c)
        print(f"  {k+' · 降风险 @mean0.6':30s}{st['sharpe']:>9.3f}{st['cagr']:>9.2%}"
              f"{st['mdd']:>9.2%}{st['calmar']:>9.3f}")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "fear_derisk_study.csv", index=False)
    print(f"\n表 -> {OUT}/fear_derisk_study.csv")
    print("\n决定规则（proposal §5）：(a) 超安慰剂 p90  (b) 打败 RV21  "
          "(c) 打败静态 B1  (d) 阴性对照 VVIX 不应同样好  (e) dev/test 同号")


if __name__ == "__main__":
    main()
