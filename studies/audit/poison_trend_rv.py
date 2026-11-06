"""未来污染测试: trend_rv_sleeve_study.weights。
把 cut 之后的价格 ×7.3 与 ×0.3, 要求 t<=cut+1 的权重逐位不变(t 日权重只用 <=t-1 的价格)。
阳性对照: 故意把权重前移 2 天(即用 t+1 信号)必须被检出, 否则测试无检出力。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import round3_common as C
import trend_rv_sleeve_study as T
from sens_rv_weights import pct_w

px = C.load_close("yahoo/QQQ.parquet")
cut = px.index[px.index.get_indexer([pd.Timestamp("2015-06-30")], method="pad")[0]]
PZ = []
for m in (7.3, 0.3):                                # 上/下两个方向都污染(纯趋势信号需要能翻转状态)
    q = px.copy(); q.loc[q.index > cut] *= m; PZ.append(q)
def W(p_, spec, e, lead=0):
    r = p_.pct_change(); pp = pct_w(r.rolling(21).std(), 504)
    w = T.weights(p_, pp, spec, e)
    return w.shift(-lead) if lead else w
lim = px.index[px.index.get_loc(cut) + 1]          # t=cut+1 的权重只用到 cut 的价格
ok = True
for spec, e in [((("lin",.5,1.5),("SMA",200)),.0), ((("lin",.0,2.),("MOM",252)),.5), ((("none",),("SMA",200)),.5), ((("lin",.5,1.5),None),1.)]:
    real = max(float(np.nanmax(np.abs(W(px, spec, e).loc[:lim].values - W(pz, spec, e).loc[:lim].values))) for pz in PZ)
    leak = max(float(np.nanmax(np.abs(W(px, spec, e, 2).loc[:lim].values - W(pz, spec, e, 2).loc[:lim].values))) for pz in PZ)
    print(f"{spec} e={e}: 真实 maxdiff={real:.2e}  泄漏对照 maxdiff={leak:.2e}")
    ok &= real == 0.0
    ok &= leak > 0
print("PASS" if ok else "FAIL")
