"""东北证券研报协议的严格复现 + 为什么我们的 Sharpe 比研报低（广度分解）+ 相关性与风格暴露。

研报协议（报告 2.1 节原文）："分五组进行回测，周频调仓，因子平滑的窗口期长度为 20，
因子回测前进行去极值、标准化以及行业市值中性化操作，最后乘以因子方向。"

本脚本按这个协议逐条实现，修掉了我之前三处偏离你已指出的问题：
  (a) 固定 h=20     -> 全期限网格 {1,5,10,20,60}，并以研报的周频(5日)为主口径
  (b) "dev 符号"     -> 改用研报的**先验方向**。研报 2.1-2.4 节明确所有因子方向为负，不是拟合出来的。
                       dev 符号 = 我之前用 2019-22 年 IC 的符号来定向，这是用了样本内信息，
                       虽然不是未来信息、但会抬高结果，而且和研报做法不同。两者都报出来对比。
  (c) 十分位多空     -> 改为**五分组**（研报的分组数），多头 = 第 5 组，多空 = 第5组 - 第1组，组内等权
中性化也补上了研报的"**市值**"维度（我之前只做了行业 + 风格）。

指标完全对齐研报表格：Rank IC、ICIR、多头年化收益、多头年化超额、多头换手率、
多空年化收益、多空年化波动率、多空 Sharpe、多空最大回撤。

广度分解：组合层的 Sharpe 近似 ICIR × sqrt(每年调仓次数)。研报在约 5000 只的全 A 股票池上
ICIR 约 0.7-1.3，所以周频下 Sharpe 约 0.7*sqrt(52)=5.2。IC 的横截面噪声标准差约 1/sqrt(N-1)，
N=98 时是 0.101，N=5000 时是 0.014，差 7.1 倍。所以**同样的真实 IC，股票池越小 ICIR 越低、
Sharpe 越低**。这是我们复现不出 Sharpe>3 的主要原因，而不是因子实现有问题。
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xs_factor_library as XL
import pdf_volume_factors as PV

OUT = Path("outputs/full/protocol"); OUT.mkdir(parents=True, exist_ok=True)
BIG = Path("data/xs_big")
SPLIT = "2023-01-03"
HS = (1, 5, 10, 20, 60)
P_REPORT = 5          # 研报的周频调仓
N_GROUP = 5           # 研报的五分组
TAU = 20
FEE_BP = 5.0
STYLE_NAMES = ["SIZE", "BETA", "VOL_21", "MOM_252", "AMIHUD", "IVOL", "BM_PROXY"]


def load_universe(which="ALL"):
    px = pd.read_parquet(BIG / "close.parquet")
    op = pd.read_parquet(BIG / "open.parquet")
    hi = pd.read_parquet(BIG / "high.parquet")
    lo = pd.read_parquet(BIG / "low.parquet")
    vo = pd.read_parquet(BIG / "volume.parquet")
    U = pd.read_csv(BIG / "universe.csv")
    cov = px.notna().mean()
    keep = cov[cov > 0.95].index
    U = U[U.ticker.isin(keep)]
    if which != "ALL":
        U = U[U.idx == which]
    cols = [c for c in U.ticker if c in px.columns]
    sec = pd.Series(U.set_index("ticker").sector.reindex(cols).fillna("Other").to_dict())
    return [x[cols].astype(float) for x in (px, op, hi, lo, vo)], sec


# ------------------------------------------------------------------ 研报的预处理
def winsor_std(df, lo=0.01, hi=0.99):
    """去极值 + 标准化（横截面）。"""
    a, b = df.quantile(lo, axis=1), df.quantile(hi, axis=1)
    w = df.clip(a, b, axis=0)
    return w.sub(w.mean(axis=1), axis=0).div(w.std(axis=1).replace(0, np.nan), axis=0)


def neutral_ind_size(df, sec: pd.Series, size: pd.DataFrame):
    """行业市值中性化：对行业哑变量 + log 市值做横截面回归，取残差（研报口径）。"""
    cols = df.columns
    secs = sorted(sec.unique())
    D = np.zeros((len(cols), len(secs)))
    for j, s in enumerate(secs):
        D[[i for i, c in enumerate(cols) if sec.get(c) == s], j] = 1.0
    Y = df.to_numpy(float)
    S = size.to_numpy(float)
    res = np.full_like(Y, np.nan)
    for t in range(len(df)):
        z = S[t]
        A = np.c_[D, z, np.ones(len(cols))]
        m = np.isfinite(z)
        if m.sum() < 30:
            continue
        y = Y[t][m]
        good = np.isfinite(y)
        if good.sum() < 30:
            continue
        Am = A[m][good]
        b, *_ = np.linalg.lstsq(Am, y[good], rcond=None)
        r = y[good] - Am @ b
        idx = np.flatnonzero(m)[good]
        res[t, idx] = r
    return pd.DataFrame(res, index=df.index, columns=cols)


# ------------------------------------------------------------------ 研报的回测
def group_books(score: pd.DataFrame, r: pd.DataFrame, P=P_REPORT, n_g=N_GROUP, fee_bp=FEE_BP):
    """五分组、周频调仓、组内等权。返回每组日收益、多空收益、多头换手。"""
    rk = score.rank(axis=1, pct=True)
    bnd = pd.Series(np.arange(len(score)) % P == 0, index=score.index)
    rets, turns = {}, {}
    for g in range(1, n_g + 1):
        lo_, hi_ = (g - 1) / n_g, g / n_g
        w = ((rk > lo_) & (rk <= hi_)).astype(float)
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
        held = (w.shift(1) if P == 1 else
                w.where(bnd, np.nan).ffill(limit=P - 1).shift(1)).fillna(0.0)
        gross = (held * r).sum(axis=1)
        cost = fee_bp * 1e-4 * held.diff().abs().sum(axis=1)
        rets[g] = gross - cost
        turns[g] = float(held.diff().abs().sum(axis=1).sum() / max((len(held) / 252), 1) / 2)
    ls = rets[n_g] - rets[1]
    return rets, ls, turns


def ann(x):
    x = x.dropna()
    if len(x) < 30:
        return np.nan
    return float((1 + x).prod() ** (252 / len(x)) - 1)


def mdd(x):
    eq = (1 + x.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def weekly_ic(score, fwd, P=P_REPORT):
    """研报的周度 Rank IC：每 P 天取一个截面，和未来 P 日收益的秩相关。ICIR = mean/std。"""
    sel = np.arange(len(score)) % P == 0
    a = score[sel].rank(axis=1)
    b = fwd[sel].rank(axis=1)
    ic = a.corrwith(b, axis=1).dropna()
    return ic


def fwd_ret(px, h):
    return np.log(px.shift(-h - 1) / px.shift(-1))


# ------------------------------------------------------------------ main
def main(which="ALL"):
    (px, op, hi, lo, vo), sec = load_universe(which)
    print(f"universe {which}: {px.shape[1]} names, {px.shape[0]} days, {sec.nunique()} sectors", flush=True)
    F0, r = XL.build(px, op, hi, lo, vo)
    size = np.log1p(px * vo.rolling(252, min_periods=60).mean())
    split = int(np.searchsorted(px.index, SPLIT))

    # 报告因子的日频可得子集 + 稳定性版本（5 分钟版本在 pdf_volume_factors，数据下载中）
    proxy_map = {"LOGV_STD": "TURN_VOL", "VOL_RATIO_1": "OVN_SHARE", "V_P_SKEWNESS": "SKEW_21",
                 "V_P_REVERSAL": "REV_1", "SIG_UP_PV_STD": "RS_UP", "OFI_PROXY": "OFI_D",
                 "AVG_SIZE_RATIO": "CLOSE_LOC"}
    rng_ = (hi - lo).replace(0, np.nan)
    locf = ((px - lo) / rng_).clip(0, 1)
    dv = px * vo
    F0["OFI_D"] = ((2 * locf - 1) * dv).rolling(TAU, min_periods=10).sum() / dv.rolling(TAU, min_periods=10).sum()
    F = {k: v for k, v in F0.items()}
    F.update({f"{k}_STAB": PV.stability(v, TAU) for k, v in F0.items()})
    F = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in F.items()}

    # ---- 研报预处理：去极值 -> 标准化 -> 行业市值中性化 -> 20 日平滑
    print("preprocessing (winsor -> standardise -> industry+size neutral -> smooth)...", flush=True)
    Z = {}
    for k, v in F.items():
        z = neutral_ind_size(winsor_std(v), sec, size)
        Z[k] = z.rolling(TAU, min_periods=TAU // 2).mean()
    rows = []
    for h in HS:
        fwd = fwd_ret(px, h)
        P = max(1, min(h, 20)) if h != P_REPORT else P_REPORT
        for k, v in Z.items():
            ic_fit = weekly_ic(v.iloc[:split], fwd.iloc[:split], P)
            dev_sign = float(np.sign(ic_fit.mean()) or 1.0)
            for sign_mode, sg in (("report(-1)", -1.0), ("dev", dev_sign)):
                sc = v * sg
                ic = weekly_ic(sc.iloc[split:], fwd.iloc[split:], P)
                rets, ls, turns = group_books(sc, r, P)
                lng = rets[N_GROUP].iloc[split:]
                lsx = ls.iloc[split:]
                eq = (1 + r.mean(axis=1).iloc[split:]).prod() ** (252 / len(lng)) - 1
                rows.append(dict(universe=which, h=h, P=P, factor=k, sign_mode=sign_mode, sign=sg,
                                 rank_ic=float(ic.mean()), icir=float(ic.mean() / ic.std()) if ic.std() else np.nan,
                                 n_ic=len(ic), long_ann=ann(lng), long_excess=ann(lng) - eq,
                                 long_turn=turns[N_GROUP], ls_ann=ann(lsx),
                                 ls_vol=float(lsx.std() * np.sqrt(252)),
                                 ls_sharpe=float(lsx.mean() / lsx.std() * np.sqrt(252)) if lsx.std() else np.nan,
                                 ls_mdd=mdd(lsx),
                                 icir_x_sqrtN=float(ic.mean() / ic.std() * np.sqrt(252 / P)) if ic.std() else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / f"protocol_{which}.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 400)

    rep = R[(R.P == P_REPORT) & (R.h == P_REPORT) & (R.sign_mode == "report(-1)")]
    print(f"\n=== 研报口径（周频、五分组、行业市值中性、先验方向、{which}）TOP 20 by 多空 Sharpe ===")
    cols = ["factor", "rank_ic", "icir", "long_ann", "long_excess", "long_turn", "ls_ann", "ls_vol",
            "ls_sharpe", "ls_mdd"]
    print(rep.nlargest(20, "ls_sharpe")[cols].round(4).to_string(index=False))

    print("\n=== 先验方向 vs dev 符号（你问的 dev 符号是什么）===")
    cmp = R[(R.P == P_REPORT) & (R.h == P_REPORT)].pivot_table(index="factor", columns="sign_mode",
                                                               values="ls_sharpe")
    cmp["diff(dev-report)"] = cmp["dev"] - cmp["report(-1)"]
    print(cmp.describe().round(3).to_string())
    print(f"  dev 符号与研报先验(-1)一致的比例: "
          f"{(R[(R.sign_mode == 'dev') & (R.h == P_REPORT)].sign < 0).mean():.1%}")

    print("\n=== 期限网格（不再固定 h=20），先验方向，均值 ===")
    print(R[R.sign_mode == "report(-1)"].groupby("h")[["rank_ic", "icir", "ls_sharpe", "ls_vol", "long_turn"]]
          .mean().round(4).to_string())

    print("\n=== 广度分解：Sharpe ≈ ICIR × sqrt(年调仓次数) ===")
    q = rep.dropna(subset=["icir", "ls_sharpe"])
    print(f"  实测 corr(ICIR×sqrt(52), 多空Sharpe) = {q.icir_x_sqrtN.corr(q.ls_sharpe):.3f}")
    print(f"  本池 N = {px.shape[1]}, IC 横截面噪声标准差 ≈ 1/sqrt(N-1) = {1/np.sqrt(px.shape[1]-1):.4f}")
    print(f"  研报池 N ≈ 5000                       ≈ {1/np.sqrt(4999):.4f}  (相差 {np.sqrt(4999/(px.shape[1]-1)):.1f} 倍)")
    print(f"  实测 ICIR 中位数 {q.icir.median():.3f} | 研报 ICIR 区间 0.22 - 1.34")

    # ---- 因子相关性与风格暴露
    last = Z[list(Z)[0]].index[-400:]
    M = pd.DataFrame({k: Z[k].loc[last].stack(future_stack=True) for k in Z}).dropna(how="all")
    C = M.corr()
    np.fill_diagonal(C.values, np.nan)
    C.to_csv(OUT / f"factor_corr_{which}.csv")
    print(f"\n=== 因子相关性（{which}，近 400 日，{len(Z)} 因子）===")
    print(f"  平均 |相关| = {C.abs().mean().mean():.3f} | 中位 = {C.abs().stack().median():.3f}")
    ev = np.linalg.eigvalsh(np.nan_to_num(C.fillna(0).values) + np.eye(len(C)))
    ev = ev[ev > 0]
    print(f"  有效独立因子数 (sum(ev))^2/sum(ev^2) = {(ev.sum() ** 2 / (ev ** 2).sum()):.1f} / {len(C)}")
    print("  研报原话：整体上因子间相关性不高")

    styles = {}
    styles["SIZE"] = size
    for s, src in [("BETA", "BETA"), ("VOL_21", "VOL_21"), ("MOM_252", "MOM_252"),
                   ("AMIHUD", "AMIHUD"), ("IVOL", "IVOL")]:
        if src in F0:
            styles[s] = F0[src]
    styles["BM_PROXY"] = -np.log(px)
    SZ = {k: winsor_std(v) for k, v in styles.items()}
    expo = {}
    for k in Z:
        e = {}
        for s, sv in SZ.items():
            a = Z[k].loc[last].stack(future_stack=True)
            b = sv.loc[last].stack(future_stack=True)
            j = pd.concat([a, b], axis=1).dropna()
            e[s] = float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) > 1000 else np.nan
        expo[k] = e
    E = pd.DataFrame(expo).T
    E.to_csv(OUT / f"style_exposure_{which}.csv")
    print(f"\n=== 风格暴露（相关系数，{which}）：研报说 略偏向流动性与特异性波动率 ===")
    print(E.abs().mean().sort_values(ascending=False).round(3).to_string())
    print("\n  暴露最大的 8 个因子（按 |AMIHUD| 流动性）：")
    print(E.reindex(E.AMIHUD.abs().sort_values(ascending=False).index).head(8).round(3).to_string())

    pd.to_pickle(dict(Z={k: v.astype("float32") for k, v in Z.items()}, r=r, sec=sec, size=size,
                      split=split, px_index=px.index, cols=list(px.columns)),
                 OUT / f"cache_{which}.pkl")
    return R, C, E


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ALL")
