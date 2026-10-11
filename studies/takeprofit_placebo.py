"""止盈的收益到底来自"择时"还是"少做空"？—— 安慰剂对照 + 叠加层口径。

必须做这个对照的理由：
  空头止盈会缩短空头在场时间，而"少做空"本身就已知有利
  （PART9：long-only 独立 Sharpe 1.185 vs 多空 0.873）。
  所以"空头止盈让 Sharpe 变好"有两种完全不同的解释：
    H1 择时：在赚到 +k·σ 时离场，抓住了日内衰竭（有信息含量）
    H2 敞口：只是空头待的时间短了／规模小了（没有信息含量，换个方式也能达到）
  区分办法是**匹配敞口、只改触发方式**：
    对照 A  tp_rand_p：每个检查时点以固定概率无条件平空（完全不看盈亏）
    对照 B  short_scale：空头规模整体缩放（完全不改时点）
  把两个对照调到与真止盈相同的"空头持仓 bar 数"，再比 Sharpe。
  若对照拿到同样的改善 -> H2 成立，止盈没有独立价值。

另外报两件之前漏掉的事：
  * 同向多次开仓的真实发生率（用户问的"能不能同天多次同向开仓"——其实本来就在发生）
  * 作为 B&H 叠加层的贡献（独立 Sharpe 不是决策指标，PART9 §3 已经定过口径）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars
from etft.intraday.strategy import buy_hold

from audit import reimpl_noise as RI
from stop_significance import block_boot_pair, sr

OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")
NPH = 30
BASE = dict(stop_long="vwap", stop_short="vwap", tail_k=1.0, tail_every_bar=True)


def chk(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def trade_stats(F, times_idx, n_days):
    """配对 fills -> 空头持仓 bar 数、同向连续开仓次数。"""
    if not len(F):
        return dict(short_bars=0.0, same_dir=0.0, n_tp=0.0, n_rand=0.0)
    F = F.copy(); F["day"] = pd.to_datetime(F.day)
    short_bars = same_dir = n_tp = n_rand = 0
    for d, g in F.groupby("day", sort=False):
        g = g.reset_index(drop=True)
        prev_sgn = None
        cur = None
        for _, r in g.iterrows():
            k = times_idx.get(r.clock)
            kind = str(r.kind)
            if kind.startswith("ENTRY"):
                sgn = 1 if r.shares > 0 else -1
                if prev_sgn is not None and sgn == prev_sgn:
                    same_dir += 1
                prev_sgn = sgn
                cur = (sgn, k)
            else:
                if kind == "TP":
                    n_tp += 1
                if kind == "RANDEXIT":
                    n_rand += 1
                if cur is not None and k is not None and cur[0] < 0:
                    short_bars += max(0, k - cur[1])
                cur = None
    return dict(short_bars=short_bars / n_days, same_dir=same_dir / n_days,
                n_tp=n_tp / n_days, n_rand=n_rand / n_days)


def main():
    b = load_bars("QQQ"); days = b.days; bh = buy_hold(b)
    times_idx = {t: i for i, t in enumerate(list(b.c.columns))}
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
        acc, st = [], []
        for off in range(NPH):
            rr, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                                reentry="hold", half_spread=0.005, commission=0.0045,
                                **BASE, **kw)
            if len(F):
                F = F.copy(); F["day"] = pd.to_datetime(F.day)
                c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
                rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
                st.append(trade_stats(F, times_idx, max(F.day.nunique(), 1)))
            acc.append(rr)
        S = pd.DataFrame(st).mean().to_dict() if st else {}
        return pd.DataFrame(acc).T.mean(axis=1), S

    cfgs = [("基线（无止盈）", {}),
            ("空头止盈 k=0.75 rearm", dict(tp_k=0.75, tp_side="short", tp_rearm="rearm"))]
    for p in (0.05, 0.10, 0.15, 0.22, 0.30):
        cfgs.append((f"对照A 随机平空 p={p}",
                     dict(tp_rand_p=p, tp_rearm="rearm")))
    for f in (0.6, 0.75, 0.9):
        cfgs.append((f"对照B 空头规模×{f}", dict(short_scale=f)))

    curves, rows = {}, []
    for name, kw in cfgs:
        r, S = run_cfg(**kw)
        curves[name] = r
        d = r.loc[DEV[0]:DEV[1]].dropna(); t = r.loc[TEST[0]:TEST[1]].dropna()
        # 叠加层：B&H + 1.0x sleeve
        ov = {}
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            x = r.loc[per[0]:per[1]].dropna()
            h = bh.reindex(x.index).dropna()
            x2 = x.reindex(h.index).fillna(0)
            ov[pl] = sr(h + x2) - sr(h)
        rows.append(dict(配置=name, 空头bar每日=S.get("short_bars", np.nan),
                         同向再开每日=S.get("same_dir", np.nan),
                         止盈次每日=S.get("n_tp", 0.0),
                         随机平空每日=S.get("n_rand", 0.0),
                         dev_sharpe=sr(d), test_sharpe=sr(t),
                         dev_叠加贡献=ov["dev"], test_叠加贡献=ov["test"]))
        print(f"  {name:26s} 空头bar/日={S.get('short_bars',float('nan')):6.1f} "
              f"同向再开/日={S.get('same_dir',float('nan')):5.3f} "
              f"dev={sr(d):+.3f} test={sr(t):+.3f} "
              f"叠加 dev={ov['dev']:+.3f} test={ov['test']:+.3f}", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "takeprofit_placebo.csv", index=False)
    print("\n" + "=" * 120)
    print(R.to_string(index=False))

    # ---------- 匹配后的对比
    base = curves["基线（无止盈）"]
    tp = curves["空头止盈 k=0.75 rearm"]
    tp_bars = R.loc[R.配置 == "空头止盈 k=0.75 rearm", "空头bar每日"].iloc[0]
    print("\n" + "=" * 120)
    print(f"真止盈把空头持仓压到 {tp_bars:.1f} bar/日"
          f"（基线 {R.loc[R.配置=='基线（无止盈）','空头bar每日'].iloc[0]:.1f}）。"
          "\n下面找「空头 bar 数最接近」的两个对照，看它们能不能复制同样的改善：")
    for pref in ("对照A", "对照B"):
        sub = R[R.配置.str.startswith(pref)].copy()
        sub["diff"] = (sub.空头bar每日 - tp_bars).abs()
        bestn = sub.sort_values("diff").iloc[0].配置
        a = curves[bestn]
        print(f"\n  最匹配的{pref}：{bestn}"
              f"（空头 bar/日 {sub.sort_values('diff').iloc[0].空头bar每日:.1f}）")
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            r1, _ = block_boot_pair(tp.loc[per[0]:per[1]].dropna(),
                                    base.loc[per[0]:per[1]].dropna(), seed=1)
            r2, dist2 = block_boot_pair(a.loc[per[0]:per[1]].dropna(),
                                        base.loc[per[0]:per[1]].dropna(), seed=2)
            # 真止盈 vs 对照 的直接配对
            r3, dist3 = block_boot_pair(tp.loc[per[0]:per[1]].dropna(),
                                        a.loc[per[0]:per[1]].dropna(), seed=3)
            lo, hi = np.percentile(dist3, [2.5, 97.5])
            print(f"    {pl:4s}: 真止盈ΔSR={r1:+.3f}  对照ΔSR={r2:+.3f}  "
                  f"|  真止盈 vs 对照 = {r3:+.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"p={float((dist3<=0).mean()):.3f}")

    # ---------- 逐年
    print("\n" + "=" * 120)
    print("逐年 Sharpe（基线 -> 空头止盈 k=0.75）")
    print(f"{'年':>6} {'基线':>9} {'止盈':>9} {'差':>9}")
    for y in range(2019, 2027):
        a = base.loc[str(y)].dropna(); c = tp.loc[str(y)].dropna()
        if len(a) < 40:
            continue
        print(f"{y:>6} {sr(a):>9.3f} {sr(c):>9.3f} {sr(c)-sr(a):>+9.3f}")
    pd.DataFrame(curves).to_pickle(OUT / "takeprofit_placebo_curves.pkl")


if __name__ == "__main__":
    main()
