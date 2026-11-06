"""GEX 对当日波动幅度的预测力,在 0DTE 时代(2022-05 每日到期上线)是否衰减 —— proposal_sleeve_anatomy H2。

这是研究库里**唯一复现成功**的 GEX 结果(gex_vol_forecast:GEX 把样本外 R² 从 0.009 提到 0.035)。
用户的问题:该结果是否因为 0DTE 的出现而衰减?

口径(与既有实验一致):GEX 滞后固定 t−1(已验证 SqueezeMetrics 的 price 列 = 当日 SPX 收盘,故无前视);
因变量 log|open→close|(QQQ,load_bars);控制 log σ_14(vol targeting 用的同一个量,daily_vol);
稳健性再控制 VIX(t−1)。HAC(5)。

**必须同时处理的混淆**(来自 sleeve_anatomy 的前置诊断,不是事后想到的):
  1. GEX 与 σ_14 / VIX 的 Spearman 相关 −0.43 / −0.48 —— 单独看斜率会把「波动水平」误当「GEX 信息」;
  2. 负 GEX 日几乎只出现在 2022(41%),2024 年为 0 —— GEX 的**取值范围**随年份剧烈变化,
     用「GEX 水平」逐年比较斜率会混入范围效应。所以同时报告**年内分位数(0~1)**版本:
     它只看『今天的 GEX 在当年处于什么位置』,与水平漂移无关。
主检验 = 两个子期(2019–2022-04 vs 2022-05–2026)控制 σ_14 后斜率之差(按月区块自助)。逐年数字只作描述。

诚实的限定:GEX 的**定义**不随时代变(公开 OI × 固定持仓假设)。如果它衰减,说明的是
『这个口径的信号』衰减,**不能**证明或否定『真实 dealer 持仓』的价值。
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import paths as _P

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars
from etft.intraday.noise import daily_vol

TAB = Path("../outputs/tables") if Path("../outputs/tables").exists() else Path("outputs/tables")
TAB.mkdir(parents=True, exist_ok=True)
GEXP = _P.RAW / "squeeze_dix_gex.parquet"
SPLIT = pd.Timestamp("2022-05-01")        # SPY/QQQ/SPX 周二周四到期上线后每日到期(2022-05-11 起 SPX 每日)


def load(b):
    days = b.days
    g = pd.read_parquet(GEXP).gex.reindex(days).ffill().shift(1)
    sd = daily_vol(b, 14)
    oc = (b.day_close / b.day_open - 1).abs()
    vix = pd.read_csv(_P.RAW / "cboe" / "VIX_History.csv", parse_dates=["DATE"]).set_index("DATE").CLOSE \
        if (_P.RAW / "cboe" / "VIX_History.csv").exists() else None
    D = pd.DataFrame({"gex": g, "sd14": sd, "oc": oc})
    if vix is not None:
        D["vix"] = vix.reindex(days).ffill().shift(1)
    D = D.dropna(subset=["gex", "sd14", "oc"])
    D = D[D.oc > 0]
    D["y"] = np.log(D.oc.clip(lower=1e-5))
    D["lsd"] = np.log(D.sd14)
    D["lg"] = np.sign(D.gex) * np.log1p(D.gex.abs() / 1e9)             # 带符号 log($bn)
    D["year"] = D.index.year
    # 年内分位数(只用『当年已发生』的信息会更严格,这里用整年分位:它只是描述尺度,不进入预测)
    D["gq"] = D.groupby("year").gex.rank(pct=True)
    return D


def slope(S, xcol, ctrl=("lsd",)):
    X = sm.add_constant(S[[xcol, *ctrl]])
    m = sm.OLS(S.y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(m.params[xcol]), float(m.tvalues[xcol]), float(m.rsquared)


def boot_slope_diff(D, xcol, ctrl, nb=3000, seed=0, L=21):
    """两子期斜率之差;按 L=21 天(约一个月)区块自助,各子期内独立重抽。"""
    rg = np.random.default_rng(seed)
    A, B = D[D.index < SPLIT], D[D.index >= SPLIT]

    def one(S):
        n = len(S); nblk = int(np.ceil(n / L)); st = rg.integers(0, max(n - L, 1), size=nblk)
        sel = np.concatenate([np.arange(s, min(s + L, n)) for s in st])[:n]
        return S.iloc[sel]
    obs = slope(B, xcol, ctrl)[0] - slope(A, xcol, ctrl)[0]
    out = np.empty(nb)
    for i in range(nb):
        out[i] = slope(one(B), xcol, ctrl)[0] - slope(one(A), xcol, ctrl)[0]
    return obs, out


def main():
    b = load_bars("QQQ")
    D = load(b)
    print(f"n={len(D)}  {D.index[0].date()} .. {D.index[-1].date()}")
    print("GEX(t−1)→log|open→close|,控制 log σ_14;gq=当年内分位数,lg=带符号 log GEX($bn)\n")

    rows = []
    print(f"{'期间':12s}{'n':>6}{'neg-GEX占比':>12}{'斜率(lg)':>10}{'t':>7}{'R²':>7} | {'斜率(gq)':>9}{'t':>7} | {'单变量R²(GEX)':>13}{'单变量R²(σ14)':>14}")
    def line(name, S):
        s1, t1, r1 = slope(S, "lg"); s2, t2, r2 = slope(S, "gq")
        r_g = slope(S, "lg", ctrl=())[2]; r_s = sm.OLS(S.y, sm.add_constant(S[["lsd"]])).fit().rsquared
        print(f"{name:12s}{len(S):6d}{(S.gex<0).mean():12.1%}{s1:+10.3f}{t1:+7.1f}{r1:7.3f} | {s2:+9.3f}{t2:+7.1f} | {r_g:13.3f}{r_s:14.3f}")
        rows.append(dict(期间=name, n=len(S), neg_gex=float((S.gex < 0).mean()), slope_lg=s1, t_lg=t1, r2=r1, slope_gq=s2, t_gq=t2, r2_gex_only=r_g, r2_sd_only=r_s))
    for y in sorted(D.year.unique()):
        line(str(y), D[D.year == y])
    line("2019–2022-04", D[D.index < SPLIT]); line("2022-05–2026", D[D.index >= SPLIT]); line("全样本", D)
    pd.DataFrame(rows).to_csv(TAB / "gex_decay_by_year.csv", index=False)

    print("\n主检验(proposal H2):后期 − 前期 斜率(控制 σ_14),月区块自助 3000 次")
    for xcol, nm in (("lg", "GEX 水平(带符号 log)"), ("gq", "年内分位数(消除水平漂移)")):
        obs, dist = boot_slope_diff(D, xcol, ("lsd",))
        lo, hi = np.percentile(dist, [2.5, 97.5]); p = 2 * min((dist <= 0).mean(), (dist >= 0).mean())
        a, c = slope(D[D.index < SPLIT], xcol)[0], slope(D[D.index >= SPLIT], xcol)[0]
        print(f"  {nm:28s} 前期 {a:+.3f} → 后期 {c:+.3f};差 {obs:+.3f} [{lo:+.3f},{hi:+.3f}] 双侧 p={p:.3f};"
              f"相对变化 {obs / abs(a) * 100:+.0f}%")

    if "vix" in D.columns:
        print("\n稳健性:再控制 VIX(t−1)")
        for nm, S in (("2019–2022-04", D[D.index < SPLIT]), ("2022-05–2026", D[D.index >= SPLIT])):
            S = S.dropna(subset=["vix"]).assign(lvix=lambda x: np.log(x.vix))
            s, t, r = slope(S, "lg", ctrl=("lsd", "lvix")); s2, t2, _ = slope(S, "gq", ctrl=("lsd", "lvix"))
            print(f"  {nm:12s} lg 斜率 {s:+.3f} (t={t:+.1f}) | gq 斜率 {s2:+.3f} (t={t2:+.1f}) | R²={r:.3f}")
    return D


if __name__ == "__main__":
    main()
