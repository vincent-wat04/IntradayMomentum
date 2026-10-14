"""事件日 × Sleeve(探索性,第二轮 E 组的直接检验;日历事前已知,无拟合参数)。
对每个事件:Sleeve 日收益(30 相位均值,实盘配置)在事件日 vs 同星期几的非事件日;分 dev(2019–22)/test(2023–26);置换检验在同星期几内进行。"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import lowvol_avoidance_study as L
from event_calendar import event_frame
from etft.intraday.bars import load_bars
b = load_bars("QQQ"); days = b.days
E = event_frame(days, b.last_idx.to_numpy())
ret = pd.read_pickle(L.OUT / "sleeve_anatomy_curve.pkl"); ret.index = pd.to_datetime(ret.index); ret = ret.reindex(days).fillna(0.0)
Y = pd.read_pickle(L.OUT / "lowvol_study.pkl")["Y"]
tr = pd.Series(days.dayofweek != 0, index=days)
rg = np.random.default_rng(0)
rows = []
for c in ["e_fomc", "e_nfp", "e_opex", "e_month_start", "e_month_end", "e_half", "e_post_holiday", "e_pre_holiday", "e_fomc_eve", "e_fomc_post"]:
    for per, (a, z) in {"dev": ("2019", "2022"), "test": ("2023", "2026"), "全": ("2019", "2026")}.items():
        m = (E[c] == 1) & tr; m = m & (days >= a) & (days <= pd.Timestamp(z + "-12-31"))
        pool = tr & (days >= a) & (days <= pd.Timestamp(z + "-12-31"))
        # 同星期几对照:对每个事件日,对照 = 同星期几的非事件日
        dow = pd.Series(days.dayofweek, index=days)
        ctrl = pool & (E[c] == 0)
        x, base = ret[m], ret[ctrl]
        # 置换:在事件日所属星期几的非事件日池内随机抽同样数量的日子(分层按星期几)
        null = []
        cnt = dow[m].value_counts()
        pools = {d: np.where((ctrl & (dow == d)).to_numpy())[0] for d in cnt.index}
        rv = ret.to_numpy()
        for _ in range(3000):
            idx = np.concatenate([rg.choice(pools[d], n, replace=True) for d, n in cnt.items() if len(pools[d])])
            null.append(rv[idx].mean())
        obs = x.mean()
        # dow 匹配均值
        exp_ = sum(ret[ctrl & (dow == d)].mean() * n for d, n in cnt.items()) / cnt.sum()
        rows.append(dict(事件=c, 期间=per, 天数=int(m.sum()), 事件日均bp=obs * 1e4, 同星期对照均bp=exp_ * 1e4, 差bp=(obs - exp_) * 1e4,
                         置换p=float(np.mean(np.array(null) >= obs)) if obs >= exp_ else float(np.mean(np.array(null) <= obs)),
                         事件日SR=L.sr(x), 对照SR=L.sr(base),
                         高位移占比=float((Y.loc[m[m].index, "disp"] > Y.loc[ctrl[ctrl].index, "disp"].median()).mean())))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False)); R.to_csv(L.TAB / "event_sleeve.csv", index=False)
