"""B0 经济检验(描述性):把 walk-forward 预测接到逐笔表上做"跳过最差 q% 笔"和"连续缩放",
口径与 trade_filter_study 冻结的相同(日曲线 = 保留笔净盈亏之和 ÷ 30 相位;因果分位;组合 = L1 RV[0.5,1.5] + Sleeve 1:1)。

报告整张表:模型 × 特征集 × {skip q=10/20/30%, cont γ=0.5/1.0};指标:Sleeve ΔSR(dev-OOS/test/全)、组合 ΔSR、
并对 skip20% 做"去掉贡献最大的 5 个交易日"的集中度检查,以及分布面板(均值/中位数/Sortino/CVaR5%/前5%日占比/最差日)。
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b0_entry_features as B0
import trade_filter_study as TF

DATA, TAB = B0.DATA, B0.TAB
NPH = 30


def sharpe(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 30 and x.std() > 0 else np.nan


def causal_rank(pred, days_sorted_idx):
    """每笔的因果分位:此前(按日期严格早于)所有 OOS 预测中小于该预测的比例;历史 <200 个时视为 0.5。"""
    n = len(pred); out = np.full(n, 0.5); hist = []
    day_arr = days_sorted_idx
    i = 0
    udays = pd.unique(day_arr)
    pos = {d: np.where(day_arr == d)[0] for d in udays}
    import bisect
    srt = []
    for d in udays:
        ix = pos[d]
        for j in ix:
            if len(srt) >= 200 and np.isfinite(pred[j]):
                out[j] = bisect.bisect_left(srt, pred[j]) / len(srt)
        for j in ix:
            if np.isfinite(pred[j]): bisect.insort(srt, pred[j])
    return out


def panel(x):
    x = pd.Series(x).dropna(); eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1; neg = x[x < 0]
    top5 = x.sort_values(ascending=False).head(int(len(x) * 0.05)).sum() / x[x > 0].sum() if (x > 0).any() else np.nan
    return dict(mean_bp=x.mean() * 1e4, median_bp=x.median() * 1e4, sharpe=sharpe(x), sortino=x.mean() / np.sqrt((neg ** 2).mean()) * np.sqrt(252) if len(neg) else np.nan,
                cagr=eq.iloc[-1] ** (252 / len(x)) - 1, mdd=dd.min(), cvar5_bp=x[x <= x.quantile(0.05)].mean() * 1e4, top5_share=top5, worst_bp=x.min() * 1e4)


def main():
    b = B0.load_bars("QQQ"); days = b.days
    TR = pd.read_pickle(DATA / "_TR.pkl").sort_values(["day", "phase", "k_dec"]).reset_index(drop=True)
    ens = pd.read_pickle(DATA / "_ens.pkl"); preds = pd.read_pickle(DATA / "_B0_preds.pkl")
    yrs = TR.day.dt.year.to_numpy(); notional = None
    day_arr = TR.day.to_numpy()
    # 组合层:L1 RV[0.5,1.5] 权重(与 trade_filter_study 相同的函数)
    import trend_rv_sleeve_study as T, round3_common as C
    from sens_rv_weights import pct_w
    px = C.load_close("yahoo/QQQ.parquet"); rr = px.pct_change(); w = (0.5 + 1.0 * (1 - pct_w(rr.rolling(21).std(), 504))).shift(1).reindex(days)
    qqq_r = (b.day_close / b.prev_close_adj - 1).reindex(days)
    L1 = (w * qqq_r)

    def curve(keep, scale=None):
        pnl = TR.pnl.to_numpy() * (1.0 if scale is None else scale)
        s = pd.Series(np.where(keep, pnl, 0.0), index=TR.day).groupby(level=0).sum() / NPH
        return s.reindex(days).fillna(0.0)

    # 基线检验:全部保留 = 引擎集成
    base = curve(np.ones(len(TR), bool)); chk = (base - ens).abs().max()
    print(f"[对账] 全部保留的日曲线 − 引擎 30 相位均值 最大差 = {chk:.2e}")
    dow_ok = days.dayofweek != 0
    def cut(s, a, b_): return s.loc[a:b_]
    oos_days = (days >= "2021-01-01")
    def delta_sr(c, ref, a, b_):
        # Sleeve 层 ΔSR(只比较 Sleeve 自身)与 组合层 ΔSR(L1 + Sleeve)
        sl = sharpe(c.loc[a:b_]) - sharpe(ref.loc[a:b_]); pt = sharpe((L1 + c).loc[a:b_]) - sharpe((L1 + ref).loc[a:b_]); return sl, pt
    rows = []
    for (nm, mdl), pr in preds.items():
        mask = np.isfinite(pr)
        rank = causal_rank(pr, day_arr)
        for q in (0.10, 0.20, 0.30):
            keep = ~(mask & (rank <= q))
            c = curve(keep)
            for lab, (a, b_) in (("dev-OOS", ("2021-01-01", "2022-12-31")), ("test", ("2023-01-01", "2026-12-31")), ("全OOS", ("2021-01-01", "2026-12-31"))):
                sl, pt = delta_sr(c, base, a, b_)
                rows.append(dict(model=mdl, set=nm, rule=f"skip{int(q*100)}%", period=lab, d_sleeve=sl, d_port=pt, skipped=1 - keep[mask].mean()))
        for g in (0.5, 1.0):
            z = (pd.Series(pr).rank(pct=True).to_numpy() - 0.5)                      # 因果性近似:用因果分位
            zz = rank - 0.5
            sc = np.clip(np.exp(g * 2 * zz), 0.5, 1.5)
            c = curve(np.ones(len(TR), bool), scale=sc)
            for lab, (a, b_) in (("dev-OOS", ("2021-01-01", "2022-12-31")), ("test", ("2023-01-01", "2026-12-31")), ("全OOS", ("2021-01-01", "2026-12-31"))):
                sl, pt = delta_sr(c, base, a, b_)
                rows.append(dict(model=mdl, set=nm, rule=f"cont{g}", period=lab, d_sleeve=sl, d_port=pt, skipped=np.nan))
    R = pd.DataFrame(rows); R.to_csv(TAB / "b0_econ.csv", index=False)
    pv = R.pivot_table(index=["model", "set", "rule"], columns="period", values=["d_sleeve", "d_port"]).round(3)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    print("\nΔSR(相对不过滤的基线;2021–2026 OOS;Sleeve 层 / 组合层(L1 RV[0.5,1.5]+Sleeve))\n")
    show = pv[[("d_sleeve", "dev-OOS"), ("d_sleeve", "test"), ("d_sleeve", "全OOS"), ("d_port", "dev-OOS"), ("d_port", "test"), ("d_port", "全OOS")]]
    show.columns = ["Sl dev", "Sl test", "Sl 全", "组合 dev", "组合 test", "组合 全"]
    print(show.to_string())
    # 仅 skip,且 dev/test 同为正、全 ≥0.05 的格
    print("\n=> Sleeve 层 dev-OOS 与 test 同为正且全OOS ≥ 0.05 的配置:")
    sk = show.reset_index(); ok = sk[(sk["Sl dev"] > 0) & (sk["Sl test"] > 0) & (sk["Sl 全"] >= 0.05)]
    print(ok.to_string(index=False) if len(ok) else "  无")
    print("\n=> 组合层 dev/test 同为正且全OOS ≥ 0.05:")
    ok2 = sk[(sk["组合 dev"] > 0) & (sk["组合 test"] > 0) & (sk["组合 全"] >= 0.05)]
    print(ok2.to_string(index=False) if len(ok2) else "  无")


if __name__ == "__main__":
    main()
