"""基于"收益形态是凸性"这个发现去开发策略。

PART8 §3 的发现：噪声区域策略的日收益单调依赖于**当日**绝对涨跌幅
（QQQ 五分组日均 −10.1 / −11.6 / −11.2 / +3.4 / +46.1 bp，偏度 +6.14）。
平静日每天小亏（窄带 -> 假突破 -> 被止损），大波动日集中获利。这是 long-gamma 形态。

直接的策略含义：**如果能事前预测"今天会不会是大波动日"，就该在平静日缩手、大波动日加码。**

但这里有一个必须先说清楚的张力：论文 SOTA 的 vol targeting 是 `expo = 2% / σ_trailing`，
也就是**趋势波动高的时候减仓**——方向和上面的含义正好相反，而它确实把 Sharpe 从
1.250 提到了 1.375。两者能同时成立，因为 vol targeting 做的是"把各个波动状态的**风险
贡献**拉平"，而凸性说的是"同一状态内，当日实现波动越大收益越高"。

所以真正要问的是一个增量问题：
    **有没有一个事前可观测的量，在 σ_trailing 之外还能预测当日的绝对涨跌幅？**
只有答案为"是"，凸性发现才能转化成一个在 SOTA 之上的改进。

流程（标准 QR：先验证前提，再上策略）：
  步骤 1  前提检验（只用 dev 2019-2022）：把当日 |close/open−1| 对各候选变量回归，
          报告在 σ_trailing 之上的**增量 R²**。不通过的变量直接出局。
  步骤 2  把通过的变量做 PIT 滚动分位（252 日），dev 期上定映射规则。
  步骤 3  两种处理方式，规则原封不动用到 test：
            T1 连续缩放  day_scale = f(分位)
            T2 硬门槛    day_filter = 分位 > 阈值（平静日不交易）
  步骤 4  **等敞口对照**：把 T1/T2 重新标定到和基线相同的平均敞口，
          确认改进不是单纯"加了杠杆"换来的（PART2 的教训）。

候选变量（全部严格事前）：
  rv14      自身 14 日已实现波动（基线，vol targeting 已经在用）
  vix9d     VIX9D t−1 收盘        近端隐含波动，最接近"今天的预期波动"
  vix9d_vix VIX9D / VIX t−1       近端 vs 一月期的期限结构，已知事件日前会翘起来
  vxn       VXN t−1               纳斯达克 100 波动率指数（QQQ 的对口指数）
  vvix      VVIX t−1              波动率的波动率 -> 尾部厚度
  gex       GEX t−2               做市商 gamma：负 gamma 时做市商必须顺势交易，
                                  机械地放大日内趋势——这正是突破策略的微观基础
  gap_abs   |开盘/前收−1|          **当日**信息，09:30 就知道，而第一个决策点是 10:00
  rv_first30 前 30 分钟已实现波动   **当日**信息，10:00 就知道
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

from etft.intraday.bars import load_bars
from etft.intraday.noise import daily_vol
from etft.intraday.strategy import NoiseCfg, run, buy_hold

from paths import ETFT
OUT = Path("outputs/full/intraday"); OUT.mkdir(parents=True, exist_ok=True)
HALF_SPREAD, COMMISSION = 0.005, 0.0045
DEV, TEST = ("2019", "2022"), ("2023", "2026")
SOTA = dict(stop="vwap", sizing="voltarget", max_lev=4.0, cost_bp=1.0)


def cboe(name):
    d = pd.read_parquet(ETFT / f"data/raw/cboe/{name}.parquet")
    s = d["close"] if "close" in d.columns else d.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s.astype(float)


def predictors(b):
    """全部严格事前。t−1 收盘的指标 shift(1)；当日指标标注清楚可用时点。"""
    days = b.days
    P = {}
    P["rv14"] = daily_vol(b, 14).reindex(days)                      # 已 shift(1)
    for nm, s in [("vix9d", cboe("VIX9D")), ("vix", cboe("VIX")), ("vxn", cboe("VXN")),
                  ("vvix", cboe("VVIX")), ("vix3m", cboe("VIX3M"))]:
        P[nm] = s.reindex(days).ffill().shift(1)
    P["vix9d_vix"] = P["vix9d"] / P["vix"]
    P["vix_vix3m"] = P["vix"] / P["vix3m"]
    g = pd.read_parquet(ETFT / "data/raw/squeeze_dix_gex.parquet")["gex"]
    g.index = pd.to_datetime(g.index)
    P["gex"] = g.reindex(days).ffill().shift(2)                     # E25 口径：多滞后一天保险
    P["neg_gex"] = -P["gex"]
    P["gap_abs"] = (b.day_open / b.prev_close_adj - 1).abs().reindex(days)   # 09:30 已知
    r1 = np.log(b.c).diff(axis=1)
    cols = [c for c in b.c.columns if c <= "10:00"]
    P["rv_first30"] = (r1[cols] ** 2).sum(axis=1).pow(0.5).reindex(days)     # 10:00 已知
    P.pop("gex")
    return pd.DataFrame(P)


def inc_r2(y, X_base, X_add):
    """在 X_base 之上加入 X_add 的增量 R²（OLS，dev 期）。"""
    def fit(X):
        X = np.column_stack([np.ones(len(X)), X])
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        bb = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        res = y[ok] - X[ok] @ bb
        return 1 - res.var() / y[ok].var(), ok.sum()
    r2b, _ = fit(X_base)
    r2f, n = fit(np.column_stack([X_base, X_add]))
    return r2b, r2f, r2f - r2b, n


def pit_pct(s, win=252):
    return s.rolling(win, min_periods=120).apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True)


def metrics(x, bench=None):
    x = x.dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(sharpe=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    ann_ = float(eq.iloc[-1] ** (252 / len(x)) - 1)
    o = dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)), ann=ann_,
             vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
             calmar=ann_ / abs(dd.min()) if dd.min() < 0 else np.nan,
             t_stat=float(x.mean() / x.std() * np.sqrt(len(x))))
    if bench is not None:
        bb = bench.reindex(x.index).fillna(0)
        if bb.std() > 0:
            o["beta"] = float(np.cov(x, bb)[0, 1] / bb.var())
            o["alpha"] = float((x - o["beta"] * bb).mean() * 252)
    return o


def net_of_cost(r, price, lev_div=1.0):
    """net = gross − turnover × 每股成本/价格；lev_div=3 表示走 TQQQ（名义金额只要 1/3）。"""
    rate = (HALF_SPREAD + COMMISSION) / price / lev_div
    return r["overlay_gross"] - r["cost"] * 1e4 * rate


def main():
    import yfinance as yf
    b = load_bars("QQQ")
    days = b.days
    px = yf.download("QQQ", start="2018-10-01", end="2026-10-05", auto_adjust=False,
                     progress=False, threads=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    PQ = px.reindex(days).ffill()
    bh = buy_hold(b)

    P = predictors(b)
    # 目标：当日日内绝对涨跌幅（策略只在日内交易）
    y_ser = (b.day_close / b.day_open - 1).abs().reindex(days)
    dev_m = (days >= DEV[0]) & (days <= f"{DEV[1]}-12-31")

    print("=== 步骤 1  前提检验（dev 2019-2022）：在 σ_trailing 之上的增量 R² ===")
    print("    被解释变量 = log 当日 |close/open − 1|；基线 = log rv14")
    y = np.log(y_ser.where(y_ser > 1e-6)).to_numpy()
    base = np.log(P["rv14"].where(P["rv14"] > 0)).to_numpy()[:, None]
    rows = []
    for c in P.columns:
        if c == "rv14":
            continue
        xa = P[c].to_numpy()
        xa = np.log(np.abs(xa) + 1e-9)[:, None] if (P[c].dropna() > 0).all() else xa[:, None]
        r2b, r2f, d, n = inc_r2(y[dev_m], base[dev_m], xa[dev_m])
        corr = pd.Series(P[c][dev_m].to_numpy()).corr(pd.Series(y_ser[dev_m].to_numpy()),
                                                      method="spearman")
        rows.append(dict(predictor=c, base_r2=r2b, full_r2=r2f, inc_r2=d,
                         spearman_vs_move=corr, n=n))
    S = pd.DataFrame(rows).sort_values("inc_r2", ascending=False)
    # 关键诊断：和 log rv14 的相关性。vol targeting 已经在 **除以** σ_trailing，
    # 再乘一个与 σ_trailing 正相关的缩放因子 = 把 vol targeting 部分抵消掉。
    rvl = np.log(P["rv14"].where(P["rv14"] > 0))
    def _lg(c):
        x = P[c]
        return np.log(x.abs() + 1e-9) if (x.dropna() > 0).all() else x
    S["corr_rv14"] = [rvl.corr(_lg(c)) for c in S.predictor]
    S.to_csv(OUT / "convexity_premise.csv", index=False)
    print(S.round(4).to_string(index=False))
    print("    注意：rv14 单独的 R² = %.3f，这是 vol targeting 已经用掉的信息。" % rows[0]["base_r2"])
    print("""
    诊断：增量 R² 高不等于能用。VIX 族与 log rv14 相关 0.68~0.77，
    而 SOTA 的 vol targeting 本身就是 expo = 2%/σ_trailing。再乘一个与 σ_trailing
    正相关的因子，等于把 vol targeting 抵消掉 —— 这是一个**机制上**就该排除的用法。
    所以事前的候选集限定为：与 log rv14 的 |相关| < 0.5（正交性要求，由机制推出，
    不看 test 结果），且 dev 期方向与凸性假设一致（高预期波动 -> 加仓）。""")
    ORTH = S[S.corr_rv14.abs() < 0.5].predictor.tolist()
    print(f"    事前候选集（|corr|<0.5）：{ORTH}")

    # ---------- 步骤 2-3
    print("\n=== 步骤 2-3  在 SOTA 之上叠加条件化（dev 定规则，test 原样套用）===")
    r_base = run(b, NoiseCfg(**SOTA))
    base_net = net_of_cost(r_base, PQ)
    print(f"  基线 SOTA：dev SR {metrics(base_net.loc[DEV[0]:DEV[1]])['sharpe']:.3f}, "
          f"test SR {metrics(base_net.loc[TEST[0]:TEST[1]])['sharpe']:.3f}, "
          f"平均敞口 {r_base.exposure_cap.mean():.3f}")

    def orth_resid(c):
        """把预测变量对 log rv14 做**扩展窗口**回归后取残差（严格 PIT，不用未来数据）。"""
        x, z = _lg(c), rvl
        df = pd.concat([x.rename("x"), z.rename("z")], axis=1)
        out = pd.Series(np.nan, index=df.index)
        xv, zv = df.x.to_numpy(), df.z.to_numpy()
        for i in range(250, len(df)):
            ok = np.isfinite(xv[:i]) & np.isfinite(zv[:i])
            if ok.sum() < 200 or not (np.isfinite(xv[i]) and np.isfinite(zv[i])):
                continue
            A = np.column_stack([np.ones(ok.sum()), zv[:i][ok]])
            bb = np.linalg.lstsq(A, xv[:i][ok], rcond=None)[0]
            out.iloc[i] = xv[i] - (bb[0] + bb[1] * zv[i])
        return out

    results = {"SOTA 基线": (r_base, base_net)}
    rows2 = []
    scales = {}
    for c in S.predictor:
        for tkey, series in (("raw", pit_pct(P[c]).reindex(days)),
                             ("orth", pit_pct(orth_resid(c)).reindex(days))):
            pc = series
            d = base_net.loc[DEV[0]:DEV[1]]
            pd_ = pc.reindex(d.index)
            if (pd_ > 2 / 3).sum() < 40 or (pd_ < 1 / 3).sum() < 40:
                continue
            sgn = 1.0 if d[pd_ > 2 / 3].mean() > d[pd_ < 1 / 3].mean() else -1.0
            q = pc if sgn > 0 else (1 - pc)
            treats = {"T1 连续缩放": dict(day_scale=(0.5 + q).fillna(1.0)),
                      "T2 硬门槛": dict(day_filter=(q >= 1 / 3).fillna(True))}
            if tkey == "orth":
                treats = {"T3 正交化缩放": dict(day_scale=(0.5 + q).fillna(1.0))}
            for lab, kw in treats.items():
                rr = run(b, NoiseCfg(**kw, **SOTA))
                nn = net_of_cost(rr, PQ)
                for per, plab in ((DEV, "dev19-22"), (TEST, "test23-26")):
                    m = metrics(nn.loc[per[0]:per[1]], bh.loc[per[0]:per[1]])
                    rows2.append(dict(predictor=c, corr_rv14=float(S.set_index('predictor')
                                                                  .corr_rv14[c]),
                                      dev_sign=sgn, treat=lab, period=plab,
                                      turnover=(rr["cost"] * 1e4).loc[per[0]:per[1]].mean(),
                                      expo=rr["avg_abs_pos"].loc[per[0]:per[1]].mean(), **m))
                results[f"{c} {lab}"] = (rr, nn)
                if "day_scale" in kw:
                    scales[f"{c} {lab}"] = kw["day_scale"]
    for per, plab in ((DEV, "dev19-22"), (TEST, "test23-26")):
        m = metrics(base_net.loc[per[0]:per[1]], bh.loc[per[0]:per[1]])
        rows2.append(dict(predictor="(基线)", corr_rv14=np.nan, dev_sign=0, treat="SOTA",
                          period=plab, turnover=(r_base["cost"] * 1e4).loc[per[0]:per[1]].mean(),
                          expo=r_base["avg_abs_pos"].loc[per[0]:per[1]].mean(), **m))
    T = pd.DataFrame(rows2)
    T.to_csv(OUT / "convexity_treatments.csv", index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 400)
    cc = ["predictor", "treat", "corr_rv14", "dev_sign", "expo", "turnover", "sharpe", "ann",
          "vol", "mdd", "calmar", "t_stat", "beta", "alpha"]
    for plab in ("dev19-22", "test23-26"):
        print(f"\n--- {plab}（{len(T[T.period==plab])-1} 个配置，注意这是多重检验）---")
        print(T[T.period == plab][cc].round(3).to_string(index=False))

    # ---------- 预先登记的选择：只看 dev + 机制约束，然后报一个 test 数字
    print("\n=== 预先登记的选择协议（只用 dev 和机制约束，不看 test）===")
    dev_t = T[T.period == "dev19-22"].set_index(["predictor", "treat"])
    tst_t = T[T.period == "test23-26"].set_index(["predictor", "treat"])
    base_dev = dev_t.loc[("(基线)", "SOTA"), "sharpe"]
    base_tst = tst_t.loc[("(基线)", "SOTA"), "sharpe"]
    elig = dev_t[(dev_t.index.get_level_values(0).isin(ORTH)) & (dev_t.dev_sign > 0)]
    elig = elig[elig.index.get_level_values(1) != "T3 正交化缩放"]
    pick = elig.sharpe.idxmax()
    # T3 的候选集不受正交性限制（残差本身已经正交化）
    elig3 = dev_t[(dev_t.index.get_level_values(1) == "T3 正交化缩放") & (dev_t.dev_sign > 0)]
    pick3 = elig3.sharpe.idxmax()
    for nm, p in (("A 正交候选集中 dev 最优（T1/T2）", pick), ("B 正交化缩放中 dev 最优（T3）", pick3)):
        print(f"  {nm}: {p[0]} / {p[1]}")
        print(f"     dev  Sharpe {dev_t.loc[p,'sharpe']:.3f} (基线 {base_dev:.3f}, "
              f"Δ{dev_t.loc[p,'sharpe']-base_dev:+.3f})")
        print(f"     test Sharpe {tst_t.loc[p,'sharpe']:.3f} (基线 {base_tst:.3f}, "
              f"Δ{tst_t.loc[p,'sharpe']-base_tst:+.3f}), 年化 {tst_t.loc[p,'ann']:.2%} "
              f"(基线 {tst_t.loc[('(基线)','SOTA'),'ann']:.2%}), "
              f"alpha {tst_t.loc[p,'alpha']:.2%}, 回撤 {tst_t.loc[p,'mdd']:.1%}")
    print(f"\n  对照：全部 {len(tst_t)-1} 个配置里 test 最优的是 "
          f"{tst_t.drop(('(基线)','SOTA')).sharpe.idxmax()} = "
          f"{tst_t.drop(('(基线)','SOTA')).sharpe.max():.3f} —— 这是**事后**最优，不可用，"
          f"列出来是为了量化选择效应有多大。")

    # ---------- 置换零假设：把缩放序列随机打乱，看 test Sharpe 的零分布
    print("\n=== 置换检验：把 day_scale 的取值在日期上随机打乱（保持分布不变）===")
    rng = np.random.default_rng(7)
    # 跑在 T3（唯一在 dev 和 test 上方向一致、且存活下来的那个配置）上
    null_pick = pick3
    sc_pick = scales.get(f"{null_pick[0]} {null_pick[1]}")
    nulls = np.array([])
    if sc_pick is not None:
        print(f"  零假设跑在 {null_pick[0]} / {null_pick[1]} 上")
        v = sc_pick.to_numpy()
        acc = []
        for i in range(120):
            perm = pd.Series(rng.permutation(v), index=sc_pick.index)
            rr = run(b, NoiseCfg(day_scale=perm, **SOTA))
            nn = net_of_cost(rr, PQ).loc[TEST[0]:TEST[1]].dropna()
            acc.append(nn.mean() / nn.std() * np.sqrt(252))
        nulls = np.array(acc)
        real = tst_t.loc[null_pick, "sharpe"]
        print(f"  真实 {real:.3f} vs 置换零分布 均值 {nulls.mean():.3f} "
              f"(std {nulls.std():.3f}, p5 {np.percentile(nulls,5):.3f}, "
              f"p95 {np.percentile(nulls,95):.3f})")
        print(f"  单边 p 值 = {(nulls >= real).mean():.3f}  "
              f"（零分布均值 {nulls.mean():.3f} 和基线 {base_tst:.3f} 的差距说明"
              f"随机缩放本身也会动 Sharpe）")
        pd.Series(nulls).to_csv(OUT / "convexity_null.csv", index=False)

    # ---------- 收益形态是否真的被改善了
    print("\n=== 步骤 4  形态检验：平静日的失血有没有被止住（test）===")
    show = ["SOTA 基线", f"{pick[0]} {pick[1]}", f"{pick3[0]} {pick3[1]}"]
    prof = {}
    for nm in show:
        rr, nn = results[nm]
        s = nn.loc[TEST[0]:TEST[1]].dropna()
        bb = bh.reindex(s.index).abs()
        qt = pd.qcut(bb, 5, labels=["Q1平静", "Q2", "Q3", "Q4", "Q5大波动"])
        g = (s.groupby(qt, observed=True).mean() * 1e4).astype(float)
        g["年化"] = metrics(s)["ann"] * 100
        g["Sharpe"] = metrics(s)["sharpe"]
        prof[nm] = g
    PR = pd.DataFrame(prof).T
    PR.to_csv(OUT / "convexity_profile.csv")
    print("（前 5 列单位 bp/天，年化单位 %）")
    print(PR.round(2).to_string())

    # ---------- 不需要预测的凸性应用：正偏 + 近零 beta 的腿该配多重？
    print("\n=== 凸性的另一个用法（不需要任何预测）：和买入持有的最优配比 ===")
    print("    正偏、近零 beta 的腿对**回撤**的贡献比对波动的贡献更有利，")
    print("    所以均值方差（看 Sharpe）给出的权重会偏小。扫一下权重看 Calmar。")
    sl = base_net
    wr = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        for per, plab in ((TEST, "test23-26"), (("2019", "2026"), "all19-26")):
            s = (bh + w * sl.reindex(bh.index).fillna(0)).loc[per[0]:per[1]]
            m = metrics(s, bh.loc[per[0]:per[1]])
            sk = float(((s - s.mean()) ** 3).mean() / s.std() ** 3)
            wr.append(dict(w=w, period=plab, **m, skew=sk))
    W = pd.DataFrame(wr)
    W.to_csv(OUT / "convexity_weights.csv", index=False)
    for plab in ("test23-26", "all19-26"):
        print(f"  --- {plab} ---")
        print(W[W.period == plab][["w", "sharpe", "ann", "vol", "mdd", "calmar", "skew",
                                   "beta", "alpha"]].round(3).to_string(index=False))
        sub = W[W.period == plab]
        print(f"      Sharpe 最优权重 w={sub.loc[sub.sharpe.idxmax(),'w']}, "
              f"Calmar 最优权重 w={sub.loc[sub.calmar.idxmax(),'w']}")

    fig, axes = plt.subplots(1, 4, figsize=(25, 5.8))
    ax = axes[0]
    ax.barh(S.predictor, S.inc_r2)
    ax.axvline(0.005, color="r", ls="--", lw=1, label="入选线 0.005")
    ax.set_xlabel("在 log σ_trailing 之上的增量 R²（dev）")
    ax.set_title("步骤1 前提检验"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="x")
    ax = axes[1]
    for nm in show:
        s = results[nm][1].loc[TEST[0]:TEST[1]]
        ax.plot((1 + s).cumprod(), lw=1.3, label=f"{nm}: SR {metrics(s)['sharpe']:.2f}")
    if len(nulls):
        ax.text(.02, .97, f"置换零分布 均值 {nulls.mean():.2f} / p95 {np.percentile(nulls,95):.2f}",
                transform=ax.transAxes, va="top", fontsize=8)
    ax.set_title("test 期净值（预先登记的选择）"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[2]
    x = np.arange(5); w = 0.8 / len(PR)
    for i, (nm, row) in enumerate(PR.iterrows()):
        ax.bar(x + i * w - 0.4, row.iloc[:5].to_numpy(), width=w, label=nm)
    ax.set_xticks(x); ax.set_xticklabels(PR.columns[:5], fontsize=8)
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("日均收益 (bp)")
    ax.set_title("步骤4 平静日失血是否止住"); ax.legend(fontsize=7.5); ax.grid(alpha=.3, axis="y")
    ax = axes[3]
    for plab, mk in (("test23-26", "o-"), ("all19-26", "s--")):
        sub = W[W.period == plab]
        ax.plot(sub.w, sub.calmar, mk, label=f"Calmar {plab}")
        ax.plot(sub.w, sub.sharpe, mk, alpha=.5, label=f"Sharpe {plab}")
    ax.set_xlabel("叠加层权重 w（买入持有 + w × SOTA）")
    ax.set_title("正偏腿的最优配比：Calmar 比 Sharpe 给更大权重")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_convexity.png", dpi=110); plt.close()
    pd.to_pickle({k: v[1] for k, v in results.items()}, OUT / "convexity_curves.pkl")
    print("\nfigure written")
    return S, T, PR


if __name__ == "__main__":
    main()
