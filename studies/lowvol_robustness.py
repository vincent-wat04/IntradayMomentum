"""lowvol_avoidance_study 的事后稳健性(探索性,不用于选配置):skip 分位邻域扫描 + 逐年 + 被跳过日的画像。
读取 lowvol_study.pkl,不重新拟合。"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import lowvol_avoidance_study as L
S = pd.read_pickle(L.OUT / "lowvol_study.pkl"); pred, X, Y = S["pred"], S["X"], S["Y"]
b_days = X.index
ret = pd.read_pickle(L.OUT / "sleeve_anatomy_curve.pkl"); ret.index = pd.to_datetime(ret.index); ret = ret.reindex(b_days).fillna(0)
td = pd.Series(b_days.dayofweek != 0, index=b_days)
def sr(x): return L.sr(x)
print("== 邻域扫描 A·ridge[rv] skip(q):Sleeve ΔSR(dev-OOS / test / 全)")
yh = pred[("A·ridge", "rv")]
for q in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
    s = L.causal_s(yh, "skip", q); xs = ret * s
    r = [sr(L.period(xs[td], k)) - sr(L.period(ret[td], k)) for k in ("dev-OOS", "test", "全OOS")]
    print(f"q={q:.2f} 跳过{(s.loc['2021':'2026'][td.loc['2021':'2026']]==0).mean():.2f}  " + "  ".join(f"{v:+.3f}" for v in r))
print("\n== 逐年 skip0.2(A·ridge[rv]) Sleeve SR:基线 → 缩放后;被跳过日的基线日均收益(bp)")
s = L.causal_s(yh, "skip", 0.2)
for y in range(2021, 2027):
    m = (b_days.year == y) & td.to_numpy()
    sk = m & (s.to_numpy() == 0)
    print(y, f"SR {sr(ret[m]):.2f} → {sr((ret*s)[m]):.2f}   跳过{sk.sum()}天  被跳日均 {ret[sk].mean()*1e4:+.1f}bp  未跳日均 {ret[m&~sk].mean()*1e4:+.1f}bp")
m = (b_days.year >= 2021) & td.to_numpy(); sk = m & (s.to_numpy() == 0)
print(f"\n全OOS: 被跳过日 n={sk.sum()} 日均 {ret[sk].mean()*1e4:+.1f}bp 标准差 {ret[sk].std()*1e4:.1f}bp; 其余 n={(m&~sk).sum()} 日均 {ret[m&~sk].mean()*1e4:+.1f}bp 标准差 {ret[m&~sk].std()*1e4:.1f}bp")
# 被跳日是否就是『低 VIX 日』?(与单纯按 VIX9D 分位跳过对比)
for col in ("log_vix9d", "log_sd14", "log_vix"):
    v = X[col]; xs = []
    for q in (0.2,):
        rank = pd.Series(np.nan, index=b_days); hist = []
        sv = np.ones(len(v))
        for i, x in enumerate(v.to_numpy()):
            if np.isfinite(x):
                if len(hist) >= 60: sv[i] = 0.0 if np.mean(np.array(hist) < x) <= q else 1.0
                hist.append(x)
        sv = pd.Series(np.where(b_days.year >= 2021, sv, 1.0), index=b_days)
        print(f"对照:仅按 {col} 因果最低20%跳过: ΔSR dev-OOS {sr(L.period((ret*sv)[td],'dev-OOS'))-sr(L.period(ret[td],'dev-OOS')):+.3f} test {sr(L.period((ret*sv)[td],'test'))-sr(L.period(ret[td],'test')):+.3f} 全 {sr(L.period((ret*sv)[td],'全OOS'))-sr(L.period(ret[td],'全OOS')):+.3f}")
