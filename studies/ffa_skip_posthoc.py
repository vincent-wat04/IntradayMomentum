"""事后(探索性):E2『避开预测的低振幅日』的稳健性 —— 邻近阈值、置换族、组合层、振幅 vs RV 的差别。"""
import os, sys, warnings; warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS",): os.environ.setdefault(_v, "1")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import lowvol_avoidance_study as L, round3_common as C, trend_rv_sleeve_study as T, disp_pred_study as DP
from etft.intraday.bars import load_bars
from sens_rv_weights import pct_w
S = pd.read_pickle(L.OUT / "ffa_forecasts.pkl"); FC, dates, R = S["FC"], S["dates"], S["R"]
b = load_bars("QQQ"); days = b.days; yrs = days.year
base = pd.read_pickle(L.OUT / "sleeve_anatomy_curve.pkl"); base.index = pd.to_datetime(base.index); base = base.reindex(days).fillna(0.0)
td = pd.Series(days.dayofweek != 0, index=days); sel = td.to_numpy() & (yrs >= 2019)
bh = pd.read_pickle(L.OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
port = lambda sl: T.build(bh, sl.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px_all, p_all)[0]
bp = port(base); bp = bp[bp.index >= "2019-01-01"]
a = DP.Arr(b); dn = DP.targets(a, 30).dn; Y = L.build(b)[1]
Rd = R.reindex(days)
print("Spearman(当日 Parkinson 振幅, 10:00 后归一化位移 dn) =", round(spearmanr(Rd[sel], dn[sel], nan_policy="omit")[0], 3), "| Spearman(10:00 后 log RV, dn) =", round(spearmanr(Y.y_rv[sel], dn[sel], nan_policy="omit")[0], 3))
q5 = lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1,2,3,4,5])
print("Sleeve 日均 bp 按当日振幅五分位(事后):", (base[sel].groupby(q5(Rd[sel])).mean()*1e4).round(1).to_dict())
rg = np.random.default_rng(1); idx = np.where(sel)[0]; perms = [rg.permutation(len(idx)) for _ in range(2000)]; bsr = L.sr(base.iloc[idx])
rows, nulls, obs = [], {}, {}
for m in ("FFA-P", "FFA-C", "FFA-SD", "EWMA", "HAR", "CARR", "成分CARR"):
    F = pd.Series(FC[m][1], index=dates).reindex(days).where(days.year >= 2018)
    for qq in (0.1, 0.2, 0.3):
        s = L.causal_s(F, "skip", qq); sv = s.to_numpy(); x = base * s
        o = L.sr(base.iloc[idx] * sv[idx]) - bsr; k = (m, qq); obs[k] = o
        nulls[k] = np.array([L.sr(pd.Series(base.to_numpy()[idx] * sv[idx][pi])) for pi in perms]) - bsr
        xp = port(x); xp = xp[xp.index >= "2019-01-01"]; d, dist = T.boot_dsr(xp, bp, nb=1000, L=10, seed=0)
        rows.append(dict(模型=m, 跳过q=qq, 跳过比例=float((sv[idx] == 0).mean()), Sleeve_ΔSR=o, dev=L.sr(x[sel & (yrs <= 2022)]) - L.sr(base[sel & (yrs <= 2022)]), test=L.sr(x[sel & (yrs >= 2023)]) - L.sr(base[sel & (yrs >= 2023)]),
                         组合ΔSR=d, 组合lo=np.percentile(dist, 2.5), 组合hi=np.percentile(dist, 97.5), 组合年化bp=1e4*252*(xp.mean()-bp.mean()), p单=float(np.mean(nulls[k] >= o))))
fam = np.max(np.vstack(list(nulls.values())), axis=0); D = pd.DataFrame(rows); D["p族内(21)"] = [float(np.mean(fam >= obs[(m, q)])) for m, q in zip(D.模型, D.跳过q)]
pd.set_option("display.width", 230, "display.float_format", lambda x: f"{x:8.3f}"); print(D.to_string(index=False)); D.to_csv(L.TAB / "ffa_skip_posthoc.csv", index=False)
s = L.causal_s(pd.Series(FC["FFA-C"][1], index=dates).reindex(days).where(days.year >= 2018), "skip", 0.2)
x = base * s; print("FFA-C skip20% 分年 Sleeve ΔSR:", {int(y): round(L.sr(x[sel & (yrs == y)]) - L.sr(base[sel & (yrs == y)]), 2) for y in range(2019, 2027)})
