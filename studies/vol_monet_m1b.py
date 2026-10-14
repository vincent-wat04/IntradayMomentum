"""M1 的诊断补充(事后,看到神谕缩放为负之后追加,见 PROPOSAL_vol_monetization.md 偏离记录):
Sleeve 到底靠什么赚钱——realized variance 还是『位移 |10:00→收盘|』?模型能不能预测后者?"""
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowvol_avoidance_study as L
import round3_common as C
import trend_rv_sleeve_study as T
from event_calendar import event_frame
from etft.intraday.bars import load_bars
from sens_rv_weights import pct_w


def main():
    b = load_bars("QQQ"); days = b.days
    X, Y, A, B = L.build(b); E = event_frame(days, b.last_idx.to_numpy()); EC = list(E.columns); X = pd.concat([X, E], axis=1)
    ret = pd.read_pickle(L.OUT / "sleeve_anatomy_curve.pkl"); ret.index = pd.to_datetime(ret.index); ret = ret.reindex(days).fillna(0.0)
    td = pd.Series(days.dayofweek != 0, index=days)
    pred, base_rv, _ = L.walk_forward(X, Y, {"A+B+E": (A + B + EC, "ridge"), "A": (A, "ridge")})
    Yd = Y.copy(); Yd["y_rv"] = np.log(Y.disp + 5e-4)
    predd, based, _ = L.walk_forward(X, Yd, {"A+B+E": (A + B + EC, "ridge")})
    oos = X.index[(X.index.year >= 2021) & X[A + B + EC].notna().all(1) & Y.notna().all(1) & td]
    pd.set_option("display.width", 200)
    print(f"OOS(周一排除){len(oos)} 天;Spearman(真实 y_rv, 真实 disp) = {spearmanr(Y.loc[oos,'y_rv'], Y.loc[oos,'disp'])[0]:.3f}")
    print(f"模型 A+B+E 的 RV 预测 vs disp 的 Spearman = {spearmanr(pred[('A+B+E','rv')].loc[oos], Y.loc[oos,'disp'])[0]:.3f};A: {spearmanr(pred[('A','rv')].loc[oos], Y.loc[oos,'disp'])[0]:.3f}")
    r2d = 1 - ((Yd.loc[oos, 'y_rv'] - predd[('A+B+E','rv')].loc[oos]) ** 2).sum() / ((Yd.loc[oos, 'y_rv'] - based.loc[oos]) ** 2).sum()
    print(f"直接预测 log(disp)(A+B+E ridge)的 OOS R² = {r2d:.3f}(对比 log RV 的 R² ≈ 0.72)")
    # 分位表:Sleeve 日均 bp
    q = lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    tab = pd.DataFrame({"按真实 y_rv(事后)": ret.loc[oos].groupby(q(Y.loc[oos, "y_rv"])).mean() * 1e4, "按真实 disp(事后)": ret.loc[oos].groupby(q(Y.loc[oos, "disp"])).mean() * 1e4,
                        "按预测 A+B+E(事前10:00)": ret.loc[oos].groupby(q(pred[("A+B+E", "rv")].loc[oos])).mean() * 1e4, "按预测 A(t−1)": ret.loc[oos].groupby(q(pred[("A", "rv")].loc[oos])).mean() * 1e4})
    print("\nSleeve 日均收益(bp),按五分位(1=最低):"); print(tab.round(1).to_string())
    # 神谕(disp) 缩放
    bh = pd.read_pickle(L.OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    port = lambda sl: T.build(bh, sl.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px_all, p_all)[0]
    base_p = port(ret); rows = []
    orc = Y.disp.where(Y.index.isin(oos))
    for fn, yh in (("神谕(disp)", orc), ("直接预测 disp 模型", predd[("A+B+E", "rv")])):
        for kind, par in (("skip", .2), ("skip", .3)):
            s = L.causal_s(yh, kind, par); sser = s.reindex(days).fillna(1.0); xs, xp = ret * sser, port(ret * sser)
            row = dict(预报=fn, 规则=f"{kind}{par:g}", 跳过比例=float((sser[td & (days.year >= 2021)] == 0).mean()))
            for k in ("dev-OOS", "test", "全OOS"):
                row[f"Sleeve ΔSR {k}"] = L.sr(L.period(xs[td], k)) - L.sr(L.period(ret[td], k)); row[f"组合 ΔSR {k}"] = L.sr(L.period(xp, k)) - L.sr(L.period(base_p, k))
            rows.append(row)
    M = pd.DataFrame(rows); print("\n位移(disp)口径的缩放(诊断):"); print(M.round(3).to_string(index=False)); M.to_csv(L.TAB / "vol_monet_m1b.csv", index=False)


if __name__ == "__main__":
    main()
