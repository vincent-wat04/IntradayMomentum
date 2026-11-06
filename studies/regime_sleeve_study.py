"""L0→L2:宏观 regime 状态下 Sleeve 的表现(描述性 + 圆周平移置换)。regime 信号取自 regime_round2(t−1 收盘的分数,t 日交易)。
预先指定:状态 = A2_x(t−1) 是否 >0.5(on/off),x ∈ {B,M,C,V,all};不调阈值。"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import lowvol_avoidance_study as L
from etft.intraday.bars import load_bars
b = load_bars("QQQ"); days = b.days
ret = pd.read_pickle(L.OUT / "sleeve_anatomy_curve.pkl"); ret.index = pd.to_datetime(ret.index); ret = ret.reindex(days).fillna(0.0)
td = (days.dayofweek != 0)
V = pd.read_pickle(L.OUT / "regime_timing_r2.pkl")["V"].reindex(days).shift(1)                     # t−1 收盘形成的分数,t 日使用
rng = np.random.default_rng(0)
rows = []
for A in ("A2_B", "A2_M", "A2_C", "A2_V", "A2_all"):
    st = (V[A] > 0.5)
    for per, (a, z) in {"dev": ("2019", "2022"), "test": ("2023", "2026"), "全": ("2019", "2026")}.items():
        m = td & (days >= a) & (days <= pd.Timestamp(z + "-12-31")) & V[A].notna().to_numpy()
        on, off = ret[m & st.to_numpy()], ret[m & ~st.to_numpy()]
        d = on.mean() - off.mean()
        # 圆周平移置换:平移状态序列(保持自相关),统计量 = on−off 均值差
        sv = st.to_numpy()[m]; rv = ret.to_numpy()[m]; n = len(rv); null = []
        for _ in range(3000):
            s = np.roll(sv, int(rng.integers(60, n - 60))); null.append(rv[s].mean() - rv[~s].mean())
        null = np.array(null)
        rows.append(dict(regime=A, 期间=per, on天=len(on), off天=len(off), on日均bp=on.mean() * 1e4, off日均bp=off.mean() * 1e4, 差bp=d * 1e4,
                         onSR=L.sr(on), offSR=L.sr(off), 置换p双侧=float(np.mean(np.abs(null) >= abs(d)))))
R = pd.DataFrame(rows); pd.set_option("display.width", 200); print(R.round(3).to_string(index=False)); R.to_csv(L.TAB / "regime_sleeve.csv", index=False)
