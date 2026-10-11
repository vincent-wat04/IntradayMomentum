"""GEX 用于**波动率预测**，再传导到 vol targeting 的分母。

为什么这个用法有希望（gex_reversion_study.py §3 的结果）：
  GEX 四分位与当日已实现波动的关系极干净、单调、dev/test 一致：
    |开盘->收盘|  dev 141/104/86/60bp（高/低比值 0.424）test 101/69/69/54bp（0.533）
    日内振幅      dev 272/187/155/119bp（0.436）        test 191/137/129/107bp（0.559）
  低 GEX 日的波动是高 GEX 日的 1.9~2.4 倍。这比任何收益关系都强得多。
  而策略里**正好有一处在用波动率预测**：vol targeting 的分母 σ_daily_14（纯历史量）。

三层检验，前两层**不依赖收益显著性**（功率高得多）：
  第一层  纯预测精度：样本外 R²、MSE、QLIKE。目标是当日 |开盘->收盘|。
  第二层  风控质量：vol targeting 的本职是让已实现波动稳定。
          所以直接测"策略月度已实现波动的离散度"是否下降。
  第三层  收益：配对区块自助。**做成敞口中性**（平均敞口对齐），
          这样差异只来自"择时敞口"，不来自"平均加了杠杆"。

顺带测一个现有实现的口径问题：
  策略隔夜敞口为 0，但 vol targeting 用的 σ 是**收盘到收盘**（含隔夜）。
  用"开盘到收盘"的日内波动做分母更对口 —— 这本身可能就是个改进。
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars
from etft.intraday.noise import daily_vol

from paths import ETFT
OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")
TARGET_VOL, MAX_LEV = 0.02, 4.0


def sr(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 30 and x.std() > 0 else np.nan


def boot_pair(a, b, nb=4000, L=10, seed=0):
    idx = a.index.intersection(b.index)
    A, B = a.reindex(idx).fillna(0).to_numpy(), b.reindex(idx).fillna(0).to_numpy()
    n = len(A); nblk = int(np.ceil(n / L)); rg = np.random.default_rng(seed)
    out = np.empty(nb)
    for i in range(nb):
        st = rg.integers(0, max(n - L, 1), size=nblk)
        sel = np.concatenate([np.arange(s, min(s + L, n)) for s in st])[:n]
        out[i] = sr(A[sel]) - sr(B[sel])
    return sr(A) - sr(B), out


def expanding_ols_forecast(y, X, min_train=400):
    """扩展窗口 OLS 预测（严格 PIT）。y, X 都是 numpy。返回样本外预测序列。"""
    n = len(y)
    pred = np.full(n, np.nan)
    for i in range(min_train, n):
        m = np.isfinite(y[:i]) & np.isfinite(X[:i]).all(axis=1)
        if m.sum() < min_train * 0.6 or not np.isfinite(X[i]).all():
            continue
        A = np.column_stack([np.ones(m.sum()), X[:i][m]])
        bta, *_ = np.linalg.lstsq(A, y[:i][m], rcond=None)
        pred[i] = bta[0] + X[i] @ bta[1:]
    return pred


def main():
    b = load_bars("QQQ"); days = b.days

    # ---- 目标：当日 |开盘->收盘|（策略只做日内，这才是它关心的量）
    move = (b.day_close / b.day_open - 1).abs()
    y = np.log(move.where(move > 1e-6))

    # ---- 预测变量（全部严格事前）
    sig_cc = daily_vol(b, 14).reindex(days)                      # 现用：收盘到收盘，已 shift(1)
    sig_oc = move.rolling(14, min_periods=7).mean().shift(1)     # 口径对口：开盘到收盘
    g = pd.read_parquet(ETFT / "data/raw/squeeze_dix_gex.parquet")
    g.index = pd.to_datetime(g.index)
    GEX = g["gex"].reindex(days).ffill().shift(1)                # t−1，合法最小滞后
    gz = (GEX - GEX.rolling(252, min_periods=120).mean()) / \
         GEX.rolling(252, min_periods=120).std()

    print("=" * 112)
    print("第一层  纯波动率预测精度（目标 = log |当日开盘->收盘|，扩展窗口样本外）")
    print("=" * 112)
    specs = [
        ("① log σ_cc（现用，收盘到收盘）", [np.log(sig_cc.where(sig_cc > 0))]),
        ("② log σ_oc（口径对口，开盘到收盘）", [np.log(sig_oc.where(sig_oc > 0))]),
        ("③ ② + GEX z", [np.log(sig_oc.where(sig_oc > 0)), gz]),
        ("④ ① + GEX z", [np.log(sig_cc.where(sig_cc > 0)), gz]),
        ("⑤ ① + ② + GEX z", [np.log(sig_cc.where(sig_cc > 0)),
                             np.log(sig_oc.where(sig_oc > 0)), gz]),
    ]
    yv = y.to_numpy()
    preds = {}
    print(f"   {'模型':34s} | {'dev R²':>8} {'dev MSE':>9} | {'test R²':>8} "
          f"{'test MSE':>9} {'test QLIKE':>11}")
    for nm, xs in specs:
        X = np.column_stack([v.to_numpy() for v in xs])
        p = expanding_ols_forecast(yv, X)
        preds[nm] = pd.Series(p, index=days)
        line = f"   {nm:34s} |"
        for per in (DEV, TEST):
            s = slice(per[0], f"{per[1]}-12-31")
            yy = y.loc[s].to_numpy(); pp = preds[nm].loc[s].to_numpy()
            m = np.isfinite(yy) & np.isfinite(pp)
            if m.sum() < 50:
                line += f" {'n/a':>8} {'n/a':>9} |"; continue
            r2 = 1 - ((yy[m] - pp[m]) ** 2).sum() / ((yy[m] - yy[m].mean()) ** 2).sum()
            mse = ((yy[m] - pp[m]) ** 2).mean()
            line += f" {r2:+8.4f} {mse:9.4f} |"
            if per is TEST:
                # QLIKE（对方差的常用损失）：用 exp 还原到波动尺度
                sg2 = np.exp(2 * pp[m]); rv2 = np.exp(2 * yy[m])
                ql = (rv2 / sg2 - np.log(rv2 / sg2) - 1).mean()
                line += f" {ql:11.4f}"
        print(line, flush=True)

    print("\n   注：R² 是对 log|move| 的样本外 R²（扩展窗口，min_train=400）。")
    print("   ① 是现在 vol targeting 在用的量；② 只是换成口径对口的日内波动；")
    print("   ③/④ 加 GEX。看 GEX 带来的**增量**。")

    # ---- 增量检验（Diebold-Mariano 风格：预测误差之差的 t 检验）
    print("\n" + "=" * 112)
    print("   预测误差之差的检验（Diebold-Mariano，HAC 未调整，仅作量级参考）")
    print("=" * 112)
    def dm(a_nm, b_nm, per):
        s = slice(per[0], f"{per[1]}-12-31")
        yy = y.loc[s].to_numpy()
        pa, pb = preds[a_nm].loc[s].to_numpy(), preds[b_nm].loc[s].to_numpy()
        m = np.isfinite(yy) & np.isfinite(pa) & np.isfinite(pb)
        d = (yy[m] - pa[m]) ** 2 - (yy[m] - pb[m]) ** 2      # a 的误差 − b 的误差
        return d.mean(), d.mean() / (d.std() / np.sqrt(len(d))), int(m.sum())
    for a_nm, b_nm, lab in ((specs[0][0], specs[1][0], "① vs ②（换口径）"),
                            (specs[1][0], specs[2][0], "② vs ③（②加GEX）"),
                            (specs[0][0], specs[3][0], "① vs ④（①加GEX）")):
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            dd, t, n = dm(a_nm, b_nm, per)
            print(f"   {lab:22s} {pl:5s} Δ均方误差 {dd:+.5f}  t={t:+6.2f}  n={n}"
                  f"   {'后者更好' if dd > 0 else '前者更好'}")

    # ================= 第二层：风控质量
    print("\n" + "=" * 112)
    print("第二层  风控质量：vol targeting 的本职是让已实现波动稳定")
    print("=" * 112)
    cur = pd.read_pickle(OUT / "stop_hard_tail_curves.pkl")
    base_ret = cur["vwap（当前基线）"]
    e_base = np.minimum(MAX_LEV, TARGET_VOL / sig_cc)            # 现用的敞口

    def make_expo(pred_log, train_slice):
        """把 log|move| 的预测还原成 σ 预测，并把平均敞口对齐到基线（敞口中性）。"""
        sg = np.exp(pred_log)
        e = np.minimum(MAX_LEV, TARGET_VOL / sg)
        k = e_base.loc[train_slice].mean() / e.loc[train_slice].mean()
        return (e * k).clip(upper=MAX_LEV)

    rows = []
    tr = slice(DEV[0], f"{DEV[1]}-12-31")
    cands = {"① 现用 σ_cc": e_base}
    for nm in (specs[1][0], specs[2][0], specs[3][0]):
        cands[nm] = make_expo(preds[nm], tr)
    for nm, e in cands.items():
        ratio = (e / e_base).replace([np.inf, -np.inf], np.nan)
        r = (base_ret * ratio).rename("r")
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            s = slice(per[0], f"{per[1]}-12-31")
            x = r.loc[s].dropna()
            mv = x.groupby(pd.Grouper(freq="ME")).std() * np.sqrt(252)
            rows.append(dict(模型=nm, 期间=pl, 平均敞口=float(e.loc[s].mean()),
                             年化波动=float(x.std() * np.sqrt(252)),
                             月波动离散度=float(mv.std()),
                             月波动离散度比=float(mv.std() / mv.mean()),
                             Sharpe=sr(x), 年化=float(x.mean() * 252),
                             最差日=float(x.min())))
    R = pd.DataFrame(rows)
    print(R.round(4).to_string(index=False))
    print("\n   「月波动离散度比」= 月度已实现波动的标准差/均值，**越小说明风控越到位**。")

    # ================= 第三层：收益（配对自助，敞口中性）
    print("\n" + "=" * 112)
    print("第三层  收益：配对区块自助（已做敞口中性，差异只来自择时敞口）")
    print("=" * 112)
    print(f"   {'模型':34s} | {'dev ΔSR':>9} {'95%区间':>20} {'p':>7} | "
          f"{'test ΔSR':>9} {'95%区间':>20} {'p':>7}")
    for nm, e in cands.items():
        if nm.startswith("①"):
            continue
        r = base_ret * (e / e_base)
        line = f"   {nm:34s} |"
        for per in (DEV, TEST):
            s = slice(per[0], f"{per[1]}-12-31")
            a = r.loc[s].dropna(); bb = base_ret.loc[s].dropna()
            real, dist = boot_pair(a, bb, seed=abs(hash(nm)) % 9999)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            line += f" {real:+9.3f} [{lo:+7.3f},{hi:+7.3f}] {float((dist<=0).mean()):7.3f} |"
        print(line, flush=True)

    R.to_csv(OUT / "gex_vol_forecast.csv", index=False)
    print(f"\n已写入 {OUT}/gex_vol_forecast.csv")


if __name__ == "__main__":
    main()
