"""隐含 vs 已实现波动的错位 + 恐慌后加仓 —— 按 docs/PROPOSAL_iv_rv_panic.md 执行。"""
from __future__ import annotations

import paths as _P
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fear_derisk_study import causal_pct, weight_from, stats, mdd, sr

OUT = Path("outputs/full/intraday")
RAW = _P.DATA
VER = "多空 band+尾0.5 无周一"
N_PL = 200


def load(path, col="close"):
    d = pd.read_parquet(RAW / path)
    s = d[col] if col in d.columns else d.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s[~s.index.duplicated(keep="last")].sort_index()


def fwd(r, h):
    """t 日收盘起、未来 h 日的累计收益（t+1..t+h）。"""
    lr = np.log1p(r)
    return np.expm1(lr[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1))


# ======================================================================
def part1():
    print("=" * 108)
    print("问题一：VXN（隐含）与 RV21（已实现）的对齐 / 错位")
    print("=" * 108)
    q = load("yahoo/QQQ.parquet")
    r = q.pct_change().dropna()
    vxn = load("cboe/VXN.parquet").reindex(r.index).ffill()
    rv = r.rolling(21).std() * np.sqrt(252) * 100           # 年化 %，与 VXN 同单位
    # 信号日 t 的值只用到 t 收盘；决策作用于 t+1 的收益
    p_rv = causal_pct(rv.loc["2009-09-14":])
    p_iv = causal_pct(vxn.loc["2009-09-14":])
    D = pd.DataFrame({"r": r, "rv": rv, "iv": vxn, "p_rv": p_rv, "p_iv": p_iv}).dropna()
    D = D.loc["2011":]
    print(f"样本 {D.index[0].date()} ~ {D.index[-1].date()}，{len(D)} 天")
    print(f"  水平相关 corr(VXN, RV21) = {D.iv.corr(D.rv):.3f}；"
          f"百分位相关 = {D.p_iv.corr(D.p_rv):.3f}")
    print(f"  VXN − RV21（隐含溢价）均值 {(D.iv-D.rv).mean():+.2f} 点，"
          f"为负的天数占比 {((D.iv-D.rv)<0).mean():.1%}")

    hi_iv, hi_rv = D.p_iv >= 0.5, D.p_rv >= 0.5
    Q = np.where(hi_iv & hi_rv, "HH 恐慌中", np.where(~hi_iv & ~hi_rv, "LL 平静",
                 np.where(hi_iv & ~hi_rv, "HL 定价未兑现", "LH 兑现未定价")))
    D["Q"] = Q
    for h in (1, 5, 21):
        D[f"f{h}"] = fwd(D.r, h)
    # 持续天数
    runs = (D.Q != D.Q.shift()).cumsum()
    dur = D.groupby(runs).Q.agg(["first", "size"])

    cur = pd.read_pickle(OUT / "strategy_comparison_curves.pkl")
    s = cur[VER]; s.index = pd.to_datetime(s.index)
    D["sleeve_next"] = s.shift(-1).reindex(D.index)

    print(f"\n  {'象限':14s}{'天数占比':>9}{'平均持续':>9}{'QQQ未来1日':>12}"
          f"{'未来5日':>10}{'未来21日':>10}{'21日 t':>8}{'sleeve次日bp':>13}{'sleeve n':>9}")
    out = {}
    for k in ("HH 恐慌中", "LL 平静", "HL 定价未兑现", "LH 兑现未定价"):
        x = D[D.Q == k]
        f21 = x.f21.dropna()
        # 21 日重叠 -> 用不重叠子样本估 t
        nov = f21.iloc[::21]
        t21 = nov.mean() / nov.std() * np.sqrt(len(nov)) if len(nov) > 2 else np.nan
        sl = x.sleeve_next.dropna()
        out[k] = x
        print(f"  {k:14s}{len(x)/len(D):>9.1%}{dur[dur['first']==k]['size'].mean():>8.1f}天"
              f"{x.f1.mean()*1e4:>10.1f}bp{x.f5.mean()*100:>9.2f}%{x.f21.mean()*100:>9.2f}%"
              f"{t21:>8.2f}{sl.mean()*1e4:>13.1f}{len(sl):>9}")
    uncond = D.f21.mean()
    print(f"  {'无条件':14s}{'100%':>9}{'':>9}{D.f1.mean()*1e4:>10.1f}bp"
          f"{D.f5.mean()*100:>9.2f}%{uncond*100:>9.2f}%")
    print("\n  P1 LH 未来21日 > HH ? ",
          f"{out['LH 兑现未定价'].f21.mean()*100:+.2f}% vs {out['HH 恐慌中'].f21.mean()*100:+.2f}%")
    print("  P2 HL 未来21日 > LL ? ",
          f"{out['HL 定价未兑现'].f21.mean()*100:+.2f}% vs {out['LL 平静'].f21.mean()*100:+.2f}%")

    # ---------- 策略层 A：QQQ 腿单独，两段
    print("\n" + "-" * 108)
    print("策略层 A：QQQ 腿单独（敞口匹配 mean w=1.0），信号 t 作用于 t+1")
    print("-" * 108)
    sig = {"RV": D.p_rv, "VXN": D.p_iv,
           "MIN": np.minimum(D.p_rv, D.p_iv), "MAX": np.maximum(D.p_rv, D.p_iv)}
    rr = D.r.shift(-1)                                      # 次日收益
    print(f"  {'变体':8s}{'期间':12s}{'Sharpe':>9}{'CAGR':>9}{'MDD':>9}{'Calmar':>9}")
    res = {}
    for per, pl in ((("2011", "2018"), "2011-2018"), (("2019", "2026"), "2019-2026")):
        base = rr.loc[per[0]:per[1]].dropna()
        st0 = stats(base)
        print(f"  {'B&H':8s}{pl:12s}{st0['sharpe']:>9.3f}{st0['cagr']:>9.2%}"
              f"{st0['mdd']:>9.2%}{st0['calmar']:>9.3f}")
        for k, p in sig.items():
            pp = pd.Series(p, index=D.index).loc[per[0]:per[1]]
            w = weight_from(pp, "derisk", 1.0)
            x = (w * rr.loc[per[0]:per[1]]).dropna()
            st = stats(x); res[(k, pl)] = st
            print(f"  {k:8s}{pl:12s}{st['sharpe']:>9.3f}{st['cagr']:>9.2%}"
                  f"{st['mdd']:>9.2%}{st['calmar']:>9.3f}")

    # 条件安慰剂：只平移 VXN 百分位
    print("\n  条件安慰剂：只循环平移 VXN 百分位、保持 RV21 对齐（200 次，全样本 2011-2026）")
    rng = np.random.default_rng(20261007)
    real_w = weight_from(pd.Series(np.minimum(D.p_rv, D.p_iv), index=D.index), "derisk", 1.0)
    real = (real_w * rr).dropna()
    rvonly = (weight_from(D.p_rv, "derisk", 1.0) * rr).dropna()
    piv = D.p_iv.to_numpy()
    nsr, nca = [], []
    for _ in range(N_PL):
        sh = int(rng.integers(126, len(piv) - 126))
        pm = np.minimum(D.p_rv.to_numpy(), np.roll(piv, sh))
        w = weight_from(pd.Series(pm, index=D.index), "derisk", 1.0)
        x = (w * rr).dropna()
        nsr.append(sr(x)); st = stats(x); nca.append(st["calmar"])
    nsr, nca = np.array(nsr), np.array(nca)
    rs, rc = sr(real), stats(real)["calmar"]
    print(f"    MIN 真实 Sharpe {rs:.3f}（纯 RV {sr(rvonly):.3f}）；"
          f"零分布 均值 {nsr.mean():.3f} p90 {np.percentile(nsr,90):.3f} -> 分位 {(nsr<rs).mean():.0%}")
    print(f"    MIN 真实 Calmar {rc:.3f}（纯 RV {stats(rvonly)['calmar']:.3f}）；"
          f"零分布 均值 {nca.mean():.3f} p90 {np.percentile(nca,90):.3f} -> 分位 {(nca<rc).mean():.0%}")

    # ---------- 策略层 B：组合
    print("\n" + "-" * 108)
    print("策略层 B：组合 = w·QQQ + 1.0·sleeve（2019-2026，敞口匹配 mean w=1.0）")
    print("-" * 108)
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna()
    bh.index = pd.to_datetime(bh.index)
    sl = s.reindex(bh.index).fillna(0.0)
    # 信号重新对齐到 bh 的日历：t 日信号作用于 t+1
    print(f"  {'变体':8s}{'Sharpe':>9}{'CAGR':>9}{'MDD':>9}{'Calmar':>9}")
    st0 = stats(bh + sl)
    print(f"  {'静态1:1':8s}{st0['sharpe']:>9.3f}{st0['cagr']:>9.2%}{st0['mdd']:>9.2%}{st0['calmar']:>9.3f}")
    for k, p in sig.items():
        pp = pd.Series(p, index=D.index).shift(1).reindex(bh.index)
        w = weight_from(pp, "derisk", 1.0)
        st = stats(w * bh + sl)
        print(f"  {k:8s}{st['sharpe']:>9.3f}{st['cagr']:>9.2%}{st['mdd']:>9.2%}{st['calmar']:>9.3f}")
    return D


# ======================================================================
def events(px, iv, tag):
    r = px.pct_change()
    p = causal_pct(iv)
    E = {}
    E["E1 VIX上穿30"] = (iv > 30) & (iv.shift(1) <= 30)
    E["E2 百分位上穿95%"] = (p > 0.95) & (p.shift(1) <= 0.95)
    E["E3 5日涨>50%"] = (iv / iv.shift(5) - 1) > 0.5
    hi10 = iv.rolling(10).max()
    was = p.rolling(10).max() > 0.90
    E["E4 退潮"] = was & (iv <= 0.85 * hi10) & ~((iv.shift(1) <= 0.85 * hi10.shift(1)) & was.shift(1, fill_value=False))
    out = {}
    for k, m in E.items():
        d = m[m.fillna(False)].index
        keep, last = [], None
        for x in d:                                     # 63 日去簇
            if last is None or (px.index.get_loc(x) - px.index.get_loc(last)) > 63:
                keep.append(x); last = x
        out[k] = keep
    return r, out


def part2():
    print("\n" + "=" * 108)
    print("问题二：恐慌后加仓（入场 = 事件**次日**收盘；VIX 比股指晚收盘 15 分钟）")
    print("=" * 108)
    for tag, pxf, ivf, start in (("标普500 + VIX", "yahoo/IDX_GSPC.parquet", "cboe/VIX.parquet", "1990"),
                                 ("QQQ + VXN", "yahoo/QQQ.parquet", "cboe/VXN.parquet", "2009-09-14")):
        px = load(pxf).loc[start:]
        iv = load(ivf).reindex(px.index).ffill()
        r, ev = events(px, iv, tag)
        print(f"\n【{tag}】{px.index[0].date()} ~ {px.index[-1].date()}")
        print(f"  {'事件':16s}{'独立次数':>8}{'持有':>6}{'事件均值':>10}{'无条件':>9}"
              f"{'超额':>9}{'胜率':>7}{'安慰剂p':>9}{'入场后最深回撤(中位/最差)':>26}")
        lr = np.log1p(r.fillna(0))
        rng = np.random.default_rng(7)
        valid_pos = np.arange(300, len(px) - 70)
        for k, ds in ev.items():
            pos = np.array([px.index.get_loc(d) + 1 for d in ds])      # 次日收盘入场
            pos = pos[pos < len(px) - 64]
            if len(pos) < 3:
                print(f"  {k:16s}{len(pos):>8}  样本过少"); continue
            for h in (5, 21, 63):
                cum = np.array([np.expm1(lr.iloc[i + 1:i + 1 + h].sum()) for i in pos])
                dd = np.array([((1 + r.iloc[i + 1:i + 1 + h]).cumprod().min() - 1) for i in pos])
                allc = np.expm1(lr.rolling(h).sum().shift(-h)).iloc[valid_pos].dropna()
                ex = cum.mean() - allc.mean()
                # 安慰剂：随机同样多的入场日
                nm = np.array([allc.iloc[rng.integers(0, len(allc), len(pos))].mean()
                               for _ in range(5000)])
                pv = float((nm >= cum.mean()).mean())
                print(f"  {k if h==5 else '':16s}{len(pos) if h==5 else '':>8}{h:>5}日"
                      f"{cum.mean()*100:>9.2f}%{allc.mean()*100:>8.2f}%{ex*100:>8.2f}%"
                      f"{(cum>0).mean():>7.0%}{pv:>9.3f}"
                      f"{np.median(dd)*100:>14.1f}% / {dd.min()*100:.1f}%")
        # 子样本（只对标普做，样本足够长）
        if tag.startswith("标普"):
            print("\n  子样本（21 日持有）：")
            for lo, hi in (("1990", "2008"), ("2009", "2026")):
                for k, ds in ev.items():
                    pos = np.array([px.index.get_loc(d) + 1 for d in ds
                                    if lo <= str(d.year) <= hi])
                    pos = pos[pos < len(px) - 22]
                    if len(pos) < 3:
                        print(f"    {lo}-{hi} {k:16s} n={len(pos)} 样本过少"); continue
                    cum = np.array([np.expm1(lr.iloc[i + 1:i + 22].sum()) for i in pos])
                    seg = lr.loc[lo:hi]
                    allc = np.expm1(seg.rolling(21).sum().shift(-21)).dropna()
                    print(f"    {lo}-{hi} {k:16s} n={len(pos):>3}  均值 {cum.mean()*100:+6.2f}%  "
                          f"无条件 {allc.mean()*100:+5.2f}%  超额 {(cum.mean()-allc.mean())*100:+6.2f}%  "
                          f"胜率 {(cum>0).mean():.0%}")


if __name__ == "__main__":
    part1()
    part2()
