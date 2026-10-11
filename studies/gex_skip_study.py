"""规避极高 GEX 日是否有 edge —— 严格按 docs/PROPOSAL_gex_skip.md 执行。

执行顺序刻意与 proposal 的预测顺序一致（先机制、再单调性、最后策略），
因为如果 P1 在交易层看不见，策略层的"赢"几乎必然是噪声。

不可变更项（proposal §3/§4）：
  GEX 滞后固定 t−1；阈值 = dev 期分位 80/90/95%；镜像对照 = 低 20/10/5%；
  主指标 = test 期叠加 ΔSharpe；噪声护栏 0.15；配对区块自助 10 天 × 4000。
"""
from __future__ import annotations

import paths as _P
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars
from etft.intraday.strategy import buy_hold

from audit import reimpl_noise as RI
from stop_significance import block_boot_pair, sr

OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")
NPH = 30
N_PLACEBO = 200
GUARD = 0.15
GEXP = (_P.RAW / "squeeze_dix_gex.parquet")
# 实盘配置（proposal §3）
BASE = dict(stop_long="band+tail", stop_short="band+tail", tail_k=0.5,
            tail_every_bar=True, skip_dows=(0,))
HI_Q = (0.80, 0.90, 0.95)          # 预先指定，禁止改
LO_Q = (0.20, 0.10, 0.05)          # 镜像对照


def chk(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def decomp(x, bench):
    """返回 (独立SR, β, α年化, α的HAC t, alpha SR, 叠加ΔSharpe)。"""
    idx = x.index.intersection(bench.index)
    y = x.reindex(idx).fillna(0.0); bb = bench.reindex(idx).fillna(0.0)
    X = sm.add_constant(bb.to_numpy())
    m = sm.OLS(y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    a_d, beta = float(m.params[0]), float(m.params[1])
    resid = y.to_numpy() - X @ m.params
    sd_e = float(np.std(resid, ddof=2))
    return dict(sharpe=sr(y), beta=beta, alpha=a_d * 252, t_alpha=float(m.tvalues[0]),
                alpha_sr=(a_d / sd_e * np.sqrt(252) if sd_e > 0 else np.nan),
                overlay=sr(bb + y) - sr(bb))


def main():
    b = load_bars("QQQ"); days = b.days; bh = buy_hold(b)
    G = pd.read_parquet(GEXP)
    g = G.gex.reindex(days).ffill().shift(1)        # t−1，固定
    gdev = g.loc[DEV[0]:DEV[1]].dropna()

    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-05", auto_adjust=False,
                     progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt
    pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None)
    fsp = pd.Series(1.0, index=pt.index)
    for d, r in sp.items():
        fsp[pt.index < d] *= r
    PT = (pt * fsp).reindex(days).ffill()
    scale = (b.day_close / PT) / 3.0

    def run_cfg(**kw):
        acc, fills = [], []
        for off in range(NPH):
            rr, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                                reentry="hold", half_spread=0.005, commission=0.0045,
                                **BASE, **kw)
            if len(F):
                F = F.copy(); F["day"] = pd.to_datetime(F.day)
                c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
                rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
                fills.append(F)
            acc.append(rr)
        return pd.DataFrame(acc).T.mean(axis=1), fills

    print("=" * 112)
    print("按 docs/PROPOSAL_gex_skip.md 执行。GEX 滞后 t−1；阈值用 dev 期分位。")
    print("=" * 112)
    base_ret, base_fills = run_cfg()

    # ---------------- P1：机制检验（交易层，不依赖策略收益）
    print("\n【P1 机制检验】高 GEX 日的逐笔表现是否更差？（dev/test 必须同号）")
    recs = []
    times = list(b.c.columns); tidx = {t: i for i, t in enumerate(times)}
    Cv = b.c.to_numpy(); dpos = {d: i for i, d in enumerate(days)}
    sd14 = RI.daily_sigma(b.day_close, b.prev_close_adj, 14)
    for F in base_fills:
        cur = None
        for _, r in F.iterrows():
            k = tidx.get(r.clock)
            if k is None:
                continue
            if str(r.kind).startswith("ENTRY"):
                cur = dict(day=r.day, sgn=(1.0 if r.shares > 0 else -1.0),
                           sh=float(r.shares), px=float(r.price), k=k + 1,
                           cm=float(r.commission))
            elif cur is not None:
                i = dpos.get(cur["day"])
                mfe = np.nan
                if i is not None and k > cur["k"]:
                    seg = Cv[i, cur["k"]:k + 1]
                    fav = cur["sgn"] * (seg / cur["px"] - 1.0)
                    sdd = sd14.get(cur["day"], np.nan)
                    if np.isfinite(sdd) and sdd > 0 and np.isfinite(fav).any():
                        mfe = float(np.nanmax(fav)) / sdd
                net = cur["sh"] * (float(r.price) - cur["px"]) - cur["cm"] - float(r.commission)
                recs.append(dict(day=cur["day"], net=net, mfe=mfe,
                                 stopped=int(str(r.kind) in ("STOP", "TAILSTOP"))))
                cur = None
    T = pd.DataFrame(recs)
    T["gex"] = T.day.map(g.to_dict())
    T["year"] = T.day.dt.year
    T = T.dropna(subset=["gex"])
    print(f"  共 {len(T)} 笔（{NPH} 相位）")
    print(f"  {'期间':10s}{'GEX 分组':12s}{'笔数':>8}{'逐笔均值bp':>12}"
          f"{'止损率':>9}{'MFE(σ)':>9}")
    p1 = {}
    for pn, per in (("dev", DEV), ("test", TEST)):
        sub = T[(T.year >= int(per[0])) & (T.year <= int(per[1]))]
        thr = gdev.quantile(0.90)
        for lab, m in (("高(>dev90%)", sub.gex >= thr), ("其余", sub.gex < thr)):
            x = sub[m]
            print(f"  {pn:10s}{lab:12s}{len(x):>8}{x.net.mean()*1e4:>12.2f}"
                  f"{x.stopped.mean():>9.1%}{x.mfe.mean():>9.3f}")
            p1[(pn, lab)] = (x.net.mean() * 1e4, x.stopped.mean(), x.mfe.mean())
        d_net = p1[(pn, "高(>dev90%)")][0] - p1[(pn, "其余")][0]
        d_stp = p1[(pn, "高(>dev90%)")][1] - p1[(pn, "其余")][1]
        d_mfe = p1[(pn, "高(>dev90%)")][2] - p1[(pn, "其余")][2]
        print(f"  {'':10s}{'差(高−其余)':12s}{'':>8}{d_net:>12.2f}{d_stp:>9.1%}{d_mfe:>9.3f}")
        p1[pn] = (d_net, d_stp, d_mfe)

    # P1 判定必须同时满足两件事：
    #   (i) dev 与 test 同号（稳定性）
    #   (ii) 符号方向与 H1 的预测一致（正确性）
    # 只查 (i) 是我第一版的 bug —— "稳定地与预测相反"也会被判成立。
    want = (-1, +1, -1)        # H1 预测：逐笔均值差<0、止损率差>0、MFE 差<0
    consistent = all(np.sign(p1["dev"][i]) == np.sign(p1["test"][i]) for i in range(3))
    correct = all(np.sign(p1["test"][i]) == want[i] for i in range(3))
    same_sign = consistent and correct
    print(f"\n  P1 判定：dev/test 同号 {'是' if consistent else '否'}；"
          f"方向符合 H1 {'是' if correct else '**否（三个指标全部反号）**'}"
          f" -> {'成立' if same_sign else '**不成立**'}")
    print("    （H1 预测：逐笔均值差<0、止损率差>0、MFE 差<0）")
    if consistent and not correct:
        print("    注意：结果是**稳定地与 H1 相反** —— 高 GEX 日对本策略更有利。"
              "\n    这是新假设的来源，不是 H1 的证据；不得据此采纳反向策略"
              "（见 proposal §10 的说明）。")

    # ---------------- 计算捷径：先验证，再使用
    # 策略日内平仓、无跨日状态（σ 只用历史价格，与当天是否交易无关），
    # 所以"剔除某天"应当**精确等价于**把该天收益置零。
    # 验证掉它，之后 3 阈值 + 600 次安慰剂就不必各跑 30 次模拟
    # （否则是 18,000 次 × 1.5s ≈ 7.5 小时）。
    def by_zeroing(skipset):
        r = base_ret.copy()
        idx = [d for d in r.index if pd.Timestamp(d) in skipset]
        r.loc[idx] = 0.0
        return r

    _thr = float(gdev.quantile(0.90))
    _sk = set(pd.Timestamp(d) for d in days[(g >= _thr).reindex(days).fillna(False).to_numpy()])
    _sim, _ = run_cfg(skip_days=_sk)
    _zer = by_zeroing(_sk)
    _d = (_sim.fillna(0) - _zer.fillna(0)).abs().max()
    print(f"\n【捷径验证】重跑模拟 vs 直接置零：最大差 {_d:.2e} "
          f"-> {'等价，可用捷径' if _d < 1e-12 else '**不等价，必须重跑**'}")
    if _d >= 1e-12:
        raise SystemExit("置零捷径不成立，停止：需要逐配置重跑模拟")

    # ---------------- P2/P3：策略层 + 对照
    print("\n【P2/P3 策略层】剔除高 GEX 日 vs 基线，以及镜像与随机安慰剂")
    rows = []
    for q in HI_Q:
        thr = float(gdev.quantile(q))
        sk = set(pd.Timestamp(d) for d in days[(g >= thr).reindex(days).fillna(False).to_numpy()])
        r = by_zeroing(sk)
        rec = dict(组=f"剔除高GEX >{q:.0%}", n_skip=len(sk))
        for pn, per in (("dev", DEV), ("test", TEST), ("全", None)):
            x = r if per is None else r.loc[per[0]:per[1]]
            hb = bh if per is None else bh.loc[per[0]:per[1]]
            d = decomp(x.dropna(), hb.dropna())
            rec[f"{pn}_SR"] = d["sharpe"]; rec[f"{pn}_叠加"] = d["overlay"]
            rec[f"{pn}_α"] = d["alpha"]; rec[f"{pn}_αSR"] = d["alpha_sr"]
            rec[f"{pn}_β"] = d["beta"]
        rows.append((rec, r, sk))
    for q in LO_Q:
        thr = float(gdev.quantile(q))
        sk = set(pd.Timestamp(d) for d in days[(g <= thr).reindex(days).fillna(False).to_numpy()])
        r = by_zeroing(sk)
        rec = dict(组=f"镜像·剔除低GEX <{q:.0%}", n_skip=len(sk))
        for pn, per in (("dev", DEV), ("test", TEST), ("全", None)):
            x = r if per is None else r.loc[per[0]:per[1]]
            hb = bh if per is None else bh.loc[per[0]:per[1]]
            d = decomp(x.dropna(), hb.dropna())
            rec[f"{pn}_SR"] = d["sharpe"]; rec[f"{pn}_叠加"] = d["overlay"]
            rec[f"{pn}_α"] = d["alpha"]; rec[f"{pn}_αSR"] = d["alpha_sr"]
            rec[f"{pn}_β"] = d["beta"]
        rows.append((rec, r, sk))

    bd = decomp(base_ret.loc[DEV[0]:DEV[1]].dropna(), bh.loc[DEV[0]:DEV[1]].dropna())
    bt = decomp(base_ret.loc[TEST[0]:TEST[1]].dropna(), bh.loc[TEST[0]:TEST[1]].dropna())
    ba = decomp(base_ret.dropna(), bh.dropna())
    print(f"\n  基线：dev SR {bd['sharpe']:.3f} 叠加 {bd['overlay']:+.3f} | "
          f"test SR {bt['sharpe']:.3f} 叠加 {bt['overlay']:+.3f} | "
          f"全 SR {ba['sharpe']:.3f} 叠加 {ba['overlay']:+.3f}")
    print(f"\n  {'组':24s}{'剔天':>6}{'dev SR':>9}{'test SR':>9}"
          f"{'dev叠加':>9}{'test叠加':>10}{'Δtest叠加':>11}{'test α':>8}{'test αSR':>10}")
    for rec, r, sk in rows:
        print(f"  {rec['组']:24s}{rec['n_skip']:>6}{rec['dev_SR']:>9.3f}"
              f"{rec['test_SR']:>9.3f}{rec['dev_叠加']:>9.3f}{rec['test_叠加']:>10.3f}"
              f"{rec['test_叠加']-bt['overlay']:>11.3f}{rec['test_α']:>8.2%}"
              f"{rec['test_αSR']:>10.3f}")

    # 单调性
    dts = [rec["test_叠加"] - bt["overlay"] for rec, _, _ in rows[:3]]
    mono = (dts[0] <= dts[1] <= dts[2]) or (dts[0] >= dts[1] >= dts[2])
    print(f"\n  P2 判定：3 个阈值上的 Δtest叠加 = "
          f"{dts[0]:+.3f} / {dts[1]:+.3f} / {dts[2]:+.3f} -> "
          f"{'单调' if mono else '**非单调**'}")

    # ---------------- C1 随机剔除安慰剂
    print(f"\n【C1 随机剔除安慰剂】剔除同样天数的随机日子，{N_PLACEBO} 个种子")
    rng = np.random.default_rng(20261007)
    tradable = [d for d in days if pd.Timestamp(d).dayofweek != 0]
    for qi, q in enumerate(HI_Q):
        n = rows[qi][0]["n_skip"]
        real = rows[qi][0]["test_叠加"] - bt["overlay"]
        null = []
        hbt = bh.loc[TEST[0]:TEST[1]].dropna()
        for _ in range(N_PLACEBO):
            sk = set(pd.Timestamp(d) for d in
                     rng.choice(tradable, size=min(n, len(tradable)), replace=False))
            x = by_zeroing(sk).loc[TEST[0]:TEST[1]].dropna()
            null.append(decomp(x, hbt)["overlay"] - bt["overlay"])
        null = np.array(null)
        pct = float((null < real).mean())
        print(f"  剔除高GEX >{q:.0%}（{n} 天）：真实 Δ={real:+.3f}  "
              f"随机零分布 均值 {null.mean():+.3f} std {null.std():.3f} "
              f"p90 {np.percentile(null,90):+.3f} -> 真实值在零分布的 {pct:.0%} 分位")

    # ---------------- 配对自助
    print("\n【配对区块自助】剔除高 GEX vs 基线（p = 改进 ≤ 0 的单边概率）")
    for qi, q in enumerate(HI_Q):
        r = rows[qi][1]
        line = f"  剔除高GEX >{q:.0%}"
        for pn, per in (("dev", DEV), ("test", TEST)):
            a = r.loc[per[0]:per[1]].dropna()
            bb = base_ret.loc[per[0]:per[1]].dropna()
            real, dist = block_boot_pair(a, bb, seed=qi + 1)
            line += f" | {pn} ΔSR={real:+.3f} p={float((dist<=0).mean()):.3f}"
        print(line)

    R = pd.DataFrame([rec for rec, _, _ in rows])
    R.to_csv(OUT / "gex_skip_study.csv", index=False)
    print(f"\n表 -> {OUT}/gex_skip_study.csv")
    print("\n" + "=" * 112)
    print("决定规则（proposal §8，四条全中才采纳）：")
    print(f"  (a) P1 dev/test 同号            : {'是' if same_sign else '**否**'}")
    print(f"  (b) P2 三阈值单调               : {'是' if mono else '**否**'}")
    print(f"  (c) test 叠加 Δ>{GUARD} 且超随机 p90 : 见上")
    print("  (d) 镜像对照未拿到可比好处       : 见上表")


if __name__ == "__main__":
    main()
