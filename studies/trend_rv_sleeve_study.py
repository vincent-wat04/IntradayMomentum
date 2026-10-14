"""QQQ 腿的 RV × 趋势(含 lo=0 / 空仓)与 Sleeve 叠加 —— 严格按 docs/PROPOSAL_trend_rv_sleeve.md 执行。

组合 = q · w_t · QQQ + sw · Sleeve,   w_t = R_t × g(T_t)
  R: none | linear lo:hi (RV21 的 504 日因果百分位,combo_lib / sens_rv_weights 同一实现)
  T: none | SMA200 | MOM252  (定义照搬 systematic_trend_rv.py: SMA 为 px<SMA_n;MOM 为 px[t−21]/px[t−n]−1<0)
  g(T) = 1(向上) 或 e_off(向下),e_off ∈ {0.5, 0}
所有信号 t−1 收盘 → t 日;成本 2bp/单位|Δw|;对照 = 同平均敞口的纯静态组合。

用法:
  python trend_rv_sleeve_study.py            # 有 Sleeve(2019-2026)
  python trend_rv_sleeve_study.py --long     # 无 Sleeve,QQQ 1999-2026(含 dot com / GFC)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3_common as C
from sens_rv_weights import pct_w, w_linear

OUT = Path("outputs/full/intraday"); OUT.mkdir(parents=True, exist_ok=True)
TAB = Path("../outputs/tables") if Path("../outputs/tables").exists() else Path("outputs/tables"); TAB.mkdir(parents=True, exist_ok=True)
VER = "多空 band+尾0.5 无周一"
DEV, TEST = ("2019", "2022"), ("2023", "2026")
COST_BP = 2.0
GUARD = 0.15


def sr(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 30 and x.std() > 0 else np.nan


def cagr(x):
    x = pd.Series(x).dropna(); return float((1 + x).prod() ** (252 / len(x)) - 1)


def mdd(x):
    eq = (1 + pd.Series(x).dropna()).cumprod(); return float((eq / eq.cummax() - 1).min())


def boot_dsr(a, b, nb=4000, L=10, seed=0):
    """配对循环区块自助:SR(a) − SR(b)。"""
    idx = a.index.intersection(b.index)
    A, B = a.reindex(idx).fillna(0).to_numpy(), b.reindex(idx).fillna(0).to_numpy()
    n = len(A); nblk = int(np.ceil(n / L)); rg = np.random.default_rng(seed); out = np.empty(nb)
    for i in range(nb):
        st = rg.integers(0, max(n - L, 1), size=nblk)
        sel = np.concatenate([np.arange(s, min(s + L, n)) for s in st])[:n]
        sa, sb = A[sel], B[sel]
        out[i] = (sa.mean() / sa.std() - sb.mean() / sb.std()) * np.sqrt(252)
    return sr(a) - sr(b), out


def trend_down(px, kind, n):
    """True = 趋势向下。定义同 systematic_trend_rv.py。"""
    if kind == "SMA":
        sma = px.rolling(n).mean(); return (px < sma).where(sma.notna(), False)
    m = px.shift(21) / px.shift(n) - 1
    return (m < 0).where(m.notna(), False)


def weights(px, p, spec, e_off):
    """spec = (R, T)  R=('none',) | ('lin',lo,hi)   T=None | ('SMA',n) | ('MOM',n)。返回 t 日实际作用的权重(信号已 shift(1))。"""
    R, T = spec
    w = pd.Series(1.0, index=px.index) if R[0] == "none" else w_linear(p, R[1], R[2])
    if T is not None:
        dn = trend_down(px, T[0], T[1]).astype(float)
        w = w * (1.0 - dn * (1.0 - e_off))
    return w.shift(1)


def rlabel(R):
    return "无RV" if R[0] == "none" else f"RV[{R[1]:g},{R[2]:g}]"


def tlabel(T, e):
    return "无趋势" if T is None else f"{T[0]}{T[1]}|e={e:g}"


SPECS = []
for R in (("none",), ("lin", 0.5, 1.5), ("lin", 0.0, 1.0), ("lin", 0.0, 1.5), ("lin", 0.0, 2.0)):
    SPECS.append((R, None, 1.0))
    for T in (("SMA", 200), ("MOM", 252)):
        for e in (0.5, 0.0):
            SPECS.append((R, T, e))
NEIGH = [(("SMA", 150),), (("SMA", 250),), (("MOM", 189),)]       # proposal §5(iii) 邻近格,只用于稳健性


def build(bh, s, qs, spec, e_off, px, p, with_sleeve=True):
    R, T, e = spec
    w = weights(px, p, (R, T), e).reindex(bh.index).fillna(1.0)
    turn = w.diff().abs().fillna(0.0)
    q, sw = qs
    x = q * w * bh - COST_BP / 1e4 * q * turn + (sw * s if with_sleeve else 0.0)
    return x, w


def per(x, k):
    return x if k == "全" else x.loc[(DEV if k == "dev" else TEST)[0]:(DEV if k == "dev" else TEST)[1]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--long", action="store_true"); a = ap.parse_args()
    px_all = C.load_close("yahoo/QQQ.parquet")
    r_all = px_all.pct_change(); p_all = pct_w(r_all.rolling(21).std(), 504)
    if a.long:
        bh = r_all.loc["2001-01-01":"2026-10-01"].dropna()                # 504 天预热 + 200/252 天趋势预热
        s = None; statics = [(1.0, 0.0)]
        tag = "QQQ 1999–2026(无 Sleeve)"
    else:
        cur = pd.read_pickle(OUT / "strategy_comparison_curves.pkl")
        bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
        s = cur[VER].copy(); s.index = pd.to_datetime(s.index); s = s.reindex(bh.index).fillna(0.0)
        statics = [(1.0, 1.0), (0.6, 1.4)]
        tag = "QQQ + Sleeve 2019–2026"
    print(f"{tag};样本 {bh.index[0].date()} ~ {bh.index[-1].date()} ({len(bh)} 天);成本 {COST_BP}bp;护栏 {GUARD}")
    rows = []
    for (q, sw) in statics:
        base_x, _ = build(bh, s, (q, sw), (("none",), None, 1.0), 1.0, px_all, p_all, with_sleeve=not a.long)
        for spec in SPECS:
            x, w = build(bh, s, (q, sw), spec, spec[2], px_all, p_all, with_sleeve=not a.long)
            mw = float(w.mean()); qeq = q * mw
            ctrl = qeq * bh + ((sw * s) if not a.long else 0.0)                 # 同平均敞口的纯静态对照
            for k in ("全", "dev", "test"):
                if a.long and k != "全":
                    continue
                xx, cc, bb = per(x, k), per(ctrl, k), per(base_x, k)
                if len(xx.dropna()) < 100: continue
                d, dist = boot_dsr(xx, cc, nb=2000)
                lo, hi = np.percentile(dist, [2.5, 97.5])
                rows.append(dict(静态=f"{q:g}:{sw:g}", RV=rlabel(spec[0]), 趋势=tlabel(spec[1], spec[2]), 期间=k, 平均QQQ权重=qeq,
                                 Sharpe=sr(xx), CAGR=cagr(xx), MDD=mdd(xx), Calmar=cagr(xx) / abs(mdd(xx)),
                                 对照SR=sr(cc), 对照MDD=mdd(cc), dSR=d, dSR_lo=lo, dSR_hi=hi, dMDD_pp=(mdd(xx) - mdd(cc)) * 100,
                                 dSR_vs基线=sr(xx) - sr(bb), 换手_年=float(w.diff().abs().sum() / (len(w) / 252)) * q))
    R = pd.DataFrame(rows)
    R.to_csv(TAB / ("trend_rv_long.csv" if a.long else "trend_rv_sleeve.csv"), index=False)
    pd.set_option("display.width", 250)
    for k in (("全",) if a.long else ("全", "dev", "test")):
        print("\n" + "=" * 150); print(f"【{k}】相对『同平均敞口纯静态对照』:ΔSR、ΔMDD(pp);ΔSR 区间含 0 即噪声;|ΔSR|<{GUARD} 一律按噪声"); print("=" * 150)
        for st in R.静态.unique():
            S = R[(R.静态 == st) & (R.期间 == k)]
            if not len(S): continue
            print(f"\n-- 静态 {st} --")
            print(f"{'RV':10s}{'趋势':14s}{'均QQQ':>7}{'Sharpe':>8}{'CAGR':>8}{'MDD':>8}{'Calmar':>8} | {'ΔSR(对照)':>10}{'95%区间':>20}{'ΔMDD(pp)':>10}{'换手/年':>8}")
            for _, r in S.iterrows():
                print(f"{r.RV:10s}{r.趋势:14s}{r.平均QQQ权重:7.2f}{r.Sharpe:8.3f}{r.CAGR:8.1%}{r.MDD:8.1%}{r.Calmar:8.2f} | {r.dSR:+10.3f}  [{r.dSR_lo:+6.2f},{r.dSR_hi:+6.2f}]{r.dMDD_pp:+10.1f}{r.换手_年:8.1f}")
    return R


if __name__ == "__main__":
    main()
