"""Sleeve 解剖 —— 严格按 docs/PROPOSAL_sleeve_anatomy.md 执行。

回答三件事(详见 proposal §0):
  A. 开仓 / 止损 / 持到收盘 / 「开仓→止损→再开仓」各有多少,按方向与 GEX 分层;
  B. 强 GEX 日开仓后的方向延续性是否更好(H1);GEX 对当日波动的预测力是否随 0DTE 衰减(H2);
  C. 收益的凸性画像与「事前低波 Sharpe 高 vs 事后平静日亏钱」是否矛盾(H3,描述性)。

口径(与 gex_skip_study.py 完全一致,不另起炉灶):
  基线 = 实盘配置(多空、band+tail k=0.5 逐 bar、voltarget 4x、reentry=hold、排除周一);
  GEX 滞后固定 t−1(已验证 SqueezeMetrics 的 price 列 = 当日 SPX 收盘,故 t−1 无前视);
  GEX 档位阈值只用 dev 期分位(2019–2022);30 个相位;TQQQ 单股成本;噪声护栏 0.15。

本脚本**不改任何策略代码**:所有成交来自 audit.reimpl_noise.simulate 返回的明细 F。

用法:
  python sleeve_anatomy.py --phases 1      # 只跑相位 0(开发/审计用,快)
  python sleeve_anatomy.py                 # 30 相位(正式结果)
"""
from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

import paths as _P

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars
from etft.intraday.strategy import buy_hold

from audit import reimpl_noise as RI

OUT = Path("outputs/full/intraday"); OUT.mkdir(parents=True, exist_ok=True)
TAB = Path("../outputs/tables") if Path("../outputs/tables").exists() else Path("outputs/tables"); TAB.mkdir(parents=True, exist_ok=True)
DEV, TEST = ("2019", "2022"), ("2023", "2026")
GEXP = _P.RAW / "squeeze_dix_gex.parquet"
# 实盘配置(与 gex_skip_study.py 的 BASE 逐项相同)
BASE = dict(stop_long="band+tail", stop_short="band+tail", tail_k=0.5,
            tail_every_bar=True, skip_dows=(0,))
STOP_KINDS = ("STOP", "TAILSTOP")


def chk(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def run_phases(b, phases):
    """返回 [(相位, 日收益Series, 成交明细F)]。成本换算与 gex_skip_study.run_cfg 逐行相同。"""
    import yfinance as yf
    days = b.days
    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-12", auto_adjust=False,
                     progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt
    pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None)
    fsp = pd.Series(1.0, index=pt.index)
    for d, r in sp.items():
        fsp[pt.index < d] *= r
    scale = (b.day_close / (pt * fsp).reindex(days).ffill()) / 3.0
    out = []
    for off in phases:
        rr, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                            reentry="hold", half_spread=0.005, commission=0.0045, **BASE)
        if len(F):
            F = F.copy(); F["day"] = pd.to_datetime(F.day)
            c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
            rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
        out.append((off, rr, F))
    return out


def pair_trades(b, F, off):
    """把一个相位的成交明细配对成逐笔交易,并给每笔打上结局与路径统计。

    一笔交易 = ENTRY_* → 紧随其后的第一个离场腿(STOP / TAILSTOP / MOC / 反手时的 STOP)。
    返回列:day, side, k_in, k_out, entry_px, exit_px, exit_kind, dir_ret(入场→离场方向收益),
            dir_ret_close(入场→收盘方向收益,**与是否被止损无关**,衡量延续性),mfe, mae, seq(当日第几笔)。
    """
    if F is None or not len(F):
        return pd.DataFrame()
    times = list(b.c.columns); tidx = {t: i for i, t in enumerate(times)}
    Cv = b.c.to_numpy(); dpos = {d: i for i, d in enumerate(b.days)}
    last_idx = b.last_idx.to_numpy()            # 当日最后一根有效 bar 的位置(半日市 = 13:00 那根,不是第 390 根)
    rows = []
    for d, g in F.groupby("day", sort=False):
        i = dpos.get(d)
        if i is None:
            continue
        cur, seq = None, 0
        for _, r in g.iterrows():
            kind = str(r.kind)
            k = tidx.get(r.clock)
            if kind.startswith("ENTRY"):
                seq += 1
                cur = dict(day=d, side=(1 if r.shares > 0 else -1), k_in=(k + 1 if k is not None else np.nan),
                           entry_px=float(r.price), seq=seq, phase=off)
            elif cur is not None:
                k_out = k + 1 if (k is not None and kind != "MOC") else (int(last_idx[i]) if kind == "MOC" else np.nan)
                sgn = cur["side"]
                seg = Cv[i, int(cur["k_in"]):int(k_out) + 1] if np.isfinite(cur["k_in"]) and np.isfinite(k_out) else np.array([])
                close_px = float(b.day_close.loc[d])
                fav = sgn * (seg / cur["entry_px"] - 1.0) if len(seg) else np.array([np.nan])
                rows.append({**cur, "k_out": k_out, "exit_px": float(r.price), "exit_kind": kind,
                             "dir_ret": sgn * (float(r.price) / cur["entry_px"] - 1.0),
                             "dir_ret_close": sgn * (close_px / cur["entry_px"] - 1.0),
                             "mfe": float(np.nanmax(fav)), "mae": float(np.nanmin(fav))})
                cur = None
    return pd.DataFrame(rows)


def day_table(T):
    """每个相位、每个交易日一行:开仓次数、各类结局计数、是否出现 开仓→止损→再开仓。"""
    if not len(T):
        return pd.DataFrame()
    g = T.groupby(["phase", "day"])
    D = g.agg(n_entry=("seq", "size"),
              n_stop=("exit_kind", lambda s: int(s.isin(STOP_KINDS).sum())),
              n_moc=("exit_kind", lambda s: int((s == "MOC").sum())),
              first_side=("side", "first"), last_side=("side", "last")).reset_index()
    D["reentry_day"] = D.n_entry >= 2
    # 「止损后再开仓」:存在 seq=j 的离场是止损类,且 j 后还有 seq=j+1
    flag = {}
    for (ph, d), s in T.sort_values(["phase", "day", "seq"]).groupby(["phase", "day"]):
        ex, sd = s.exit_kind.tolist(), s.side.tolist()
        any_re, same, opp = False, False, False
        for j in range(len(s) - 1):
            if ex[j] in STOP_KINDS:
                any_re = True
                if sd[j + 1] == sd[j]: same = True
                else: opp = True
        flag[(ph, d)] = (any_re, same, opp)
    D["stop_then_reentry"] = [flag[(p, d)][0] for p, d in zip(D.phase, D.day)]
    D["reentry_same"] = [flag[(p, d)][1] for p, d in zip(D.phase, D.day)]
    D["reentry_opp"] = [flag[(p, d)][2] for p, d in zip(D.phase, D.day)]
    return D


# ------------------------------------------------------------------ 分层与统计
def load_context(b):
    """GEX(t−1)、dev 期三分位阈值、σ_14,以及事后的当日实现波动。全部口径同 gex_skip_study。"""
    days = b.days
    G = pd.read_parquet(GEXP)
    g = G.gex.reindex(days).ffill().shift(1)                       # t−1,固定
    gdev = g.loc[DEV[0]:DEV[1]].dropna()
    q1, q2 = gdev.quantile(1 / 3), gdev.quantile(2 / 3)          # 只用 dev 期
    tier = pd.Series(np.where(g <= q1, "低", np.where(g <= q2, "中", "高")), index=days).where(g.notna())
    from etft.intraday.noise import daily_vol
    ctx = pd.DataFrame({"gex": g, "tier": tier, "sd14": daily_vol(b, 14),
                        "oc": (b.day_close / b.day_open - 1).abs()})
    return ctx, (float(q1), float(q2))


def boot_diff(x, y, cluster_x, cluster_y, nb=4000, seed=0):
    """两组逐笔值的均值差,按『交易日』聚类的自助(同一天的多笔/多相位不是独立样本)。"""
    rg = np.random.default_rng(seed)
    gx = x.groupby(cluster_x).agg(["sum", "count"]); gy = y.groupby(cluster_y).agg(["sum", "count"])
    sx, cx, sy, cy = gx["sum"].to_numpy(), gx["count"].to_numpy(), gy["sum"].to_numpy(), gy["count"].to_numpy()
    obs = sx.sum() / cx.sum() - sy.sum() / cy.sum()
    out = np.empty(nb)
    for i in range(nb):
        ix, iy = rg.integers(0, len(sx), len(sx)), rg.integers(0, len(sy), len(sy))
        out[i] = sx[ix].sum() / cx[ix].sum() - sy[iy].sum() / cy[iy].sum()
    return float(obs), out


def descriptive(T, D, ctx, nph):
    """proposal §3 A–D。比例类指标无偏;不要把 30 相位的笔数当独立样本。"""
    n_trade_days = D.groupby("phase").size().mean()
    n_days_all = int(ctx.gex.notna().sum())
    elig = n_days_all - int(((ctx.index.dayofweek == 0) & ctx.gex.notna()).sum())      # 排除周一后的可交易日
    print("\n" + "=" * 100)
    print("A. 开仓(相位均值)")
    print("=" * 100)
    print(f"  可交易日(排除周一){elig};平均每相位有开仓的日子 {n_trade_days:.0f}  = {n_trade_days/elig:.1%}")
    ne = D.groupby("phase").n_entry.value_counts(normalize=True).unstack(fill_value=0).mean()
    print("  开仓日里,当日开仓次数分布:", {int(k): f"{v:.1%}" for k, v in ne.items()})
    print("\n" + "=" * 100)
    print("B. 每一笔开仓的结局(相位均值的构成比)")
    print("=" * 100)
    comp = T.groupby("phase").exit_kind.value_counts(normalize=True).unstack(fill_value=0).mean()
    cnt = T.groupby("phase").exit_kind.size().mean()
    print(f"  每相位平均 {cnt:.0f} 笔:", {k: f"{v:.1%}" for k, v in comp.items()})
    bs = T.assign(side_s=T.side.map({1: "多", -1: "空"})).groupby(["side_s", "exit_kind"]).size().unstack(fill_value=0)
    print("  按方向 × 结局(各相位合计笔数):"); print((bs.div(bs.sum(axis=1), axis=0) * 100).round(1).assign(笔数=bs.sum(axis=1)).to_string())
    print("\n" + "=" * 100)
    print("C. 『开仓→止损→再开仓』")
    print("=" * 100)
    rr = D.groupby("phase").agg(sr=("stop_then_reentry", "mean"), same=("reentry_same", "mean"), opp=("reentry_opp", "mean")).mean()
    print(f"  开仓日里出现『止损后当天再开仓』的比例 {rr['sr']:.1%}(其中同向 {rr['same']:.1%}、反向 {rr['opp']:.1%})")
    print("  注:band 止损只有 10 笔/相位,所以这里几乎全部来自 TAILSTOP 之后下一检查点信号仍成立。")
    return comp


def layered(T, ctx):
    """proposal §3 D:方向 × GEX 档 的结局与『入场到收盘』方向收益(与仓位无关)。"""
    T = T.copy(); T["tier"] = T.day.map(ctx.tier.to_dict()); T["year"] = T.day.dt.year
    T["sd14"] = T.day.map(ctx.sd14.to_dict()); T["gex"] = T.day.map(ctx.gex.to_dict())
    T["stopped"] = T.exit_kind.isin(STOP_KINDS).astype(int); T = T.dropna(subset=["tier"])
    T["period"] = np.where(T.year <= int(DEV[1]), "dev", "test")
    g = T.assign(side_s=T.side.map({1: "多", -1: "空"})).groupby(["period", "side_s", "tier"])
    L = g.agg(笔数=("stopped", "size"), 止损率=("stopped", "mean"),
              持到收盘率=("exit_kind", lambda s: (s == "MOC").mean()),
              入场到收盘_bp=("dir_ret_close", lambda s: s.mean() * 1e4),
              入场到离场_bp=("dir_ret", lambda s: s.mean() * 1e4),
              MFE_bp=("mfe", lambda s: s.mean() * 1e4), MAE_bp=("mae", lambda s: s.mean() * 1e4))
    return T, L


def h1(T, ctx, nb=4000):
    """H1:高 vs 低 GEX 档,『入场到收盘』方向收益之差(双侧);再控制 σ_daily 的回归。"""
    import statsmodels.api as sm
    print("\n" + "=" * 100)
    print("H1  强 GEX 日开仓后,方向延续性(入场→收盘 方向收益)是否不同?    [双侧;dev/test 必须同号]")
    print("=" * 100)
    res = {}
    for per in ("dev", "test", "all"):
        S = T if per == "all" else T[T.period == per]
        hi, lo = S[S.tier == "高"], S[S.tier == "低"]
        if len(hi) < 30 or len(lo) < 30:
            print(f"  {per}: 样本不足(高 {len(hi)} / 低 {len(lo)})"); continue
        obs, dist = boot_diff(hi.dir_ret_close, lo.dir_ret_close, hi.day, lo.day, nb)
        lo_, hi_ = np.percentile(dist, [2.5, 97.5])
        p = 2 * min((dist <= 0).mean(), (dist >= 0).mean())
        # 控制 σ_daily:dir_ret_close ~ 高档 + 低档 + log sd14(以中档为基准)
        X = pd.DataFrame({"hi": (S.tier == "高").astype(float), "lo": (S.tier == "低").astype(float), "lsd": np.log(S.sd14)})
        m = sm.OLS(S.dir_ret_close * 1e4, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": S.day.factorize()[0]})
        print(f"  {per:4s} 高档 n={len(hi):5d} 均值 {hi.dir_ret_close.mean()*1e4:+7.1f}bp | 低档 n={len(lo):5d} 均值 {lo.dir_ret_close.mean()*1e4:+7.1f}bp | "
              f"差 {obs*1e4:+7.1f}bp [{lo_*1e4:+.1f},{hi_*1e4:+.1f}] p={p:.3f} | 控制σ后 高档系数 {m.params['hi']:+.1f}bp (t={m.tvalues['hi']:+.2f})")
        res[per] = (obs, p, float(m.params["hi"]), float(m.tvalues["hi"]))
    if "dev" in res and "test" in res:
        same = np.sign(res["dev"][0]) == np.sign(res["test"][0])
        print(f"\n  dev/test 同号:{same}(dev {res['dev'][0]*1e4:+.1f}bp / test {res['test'][0]*1e4:+.1f}bp)")
    return res


def h1_placebo(T, nperm=2000, seed=1):
    """随机置换 GEX 档位(在『交易日』层面置换,保持每日多笔的相关结构)得到零分布。"""
    rg = np.random.default_rng(seed)
    days = T.day.drop_duplicates().to_numpy(); tiers = T.drop_duplicates("day").set_index("day").tier
    obs_hi = T[T.tier == "高"].dir_ret_close.mean() - T[T.tier == "低"].dir_ret_close.mean()
    dayidx = {d: i for i, d in enumerate(days)}; di = T.day.map(dayidx).to_numpy(); y = T.dir_ret_close.to_numpy()
    base = tiers.reindex(days).to_numpy()
    cnt = 0; null = np.empty(nperm)
    for k in range(nperm):
        sh = rg.permutation(base)[di]
        a, c = y[sh == "高"], y[sh == "低"]
        null[k] = a.mean() - c.mean()
    p = float((np.abs(null) >= abs(obs_hi)).mean())
    print(f"  置换对照(档位在交易日间随机置换 {nperm} 次):真实差 {obs_hi*1e4:+.1f}bp,零分布 sd {null.std()*1e4:.1f}bp,双侧 p={p:.3f}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", type=int, default=30)
    a = ap.parse_args()
    b = load_bars("QQQ"); days = b.days; bh = buy_hold(b)
    print(f"QQQ {len(days)} 天;相位数 {a.phases};基线 = 实盘配置;GEX 滞后 t−1,dev 阈值")
    res = run_phases(b, range(a.phases))
    curve = pd.DataFrame({o: r for o, r, _ in res}).mean(axis=1)
    Ts = [pair_trades(b, F, o) for o, _, F in res]
    T = pd.concat([t for t in Ts if len(t)], ignore_index=True)
    D = day_table(T)
    ctx, thr = load_context(b)
    print(f"GEX 三分位阈值(仅 dev 期):低≤{thr[0]/1e9:.2f}bn < 中 ≤{thr[1]/1e9:.2f}bn < 高")
    T.to_pickle(OUT / "sleeve_anatomy_trades.pkl"); D.to_pickle(OUT / "sleeve_anatomy_days.pkl"); curve.to_pickle(OUT / "sleeve_anatomy_curve.pkl")
    descriptive(T, D, ctx, a.phases)
    T2, L = layered(T, ctx)
    print("\n" + "=" * 100)
    print("D. 方向 × GEX 档 × 期间(各相位合计;『入场到收盘』= sgn×(收盘/入场−1),与仓位无关)")
    print("=" * 100)
    print(L.round(3).to_string())
    L.to_csv(TAB / "sleeve_anatomy_layered.csv")
    h1(T2, ctx); h1_placebo(T2)
    return b, T2, D, ctx, curve, bh


if __name__ == "__main__":
    main()
