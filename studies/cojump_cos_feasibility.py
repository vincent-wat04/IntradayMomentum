"""OHLC-COS 共同跳跃检测器:可行性与校准(**不使用任何收益/策略数据,不选参**)。

回答三件事(对应 docs/PROPOSAL_cojump_features.md §3):
  C1 校准:把配对资产整体错开一个交易日(同一分钟、不同日 -> 构造上没有共同跳跃)作零分布。
        真实配对的检出率 vs 错开配对的检出率;日度误报率。
  C2 功效/特异性:在真实噪声里人工注入一根"光头光脚"K线——(i)两个资产同向同时注入(共同跳跃),
        (ii)只注入一个资产(特质跳跃)。检出率对幅度的曲线。
  C3 事件命中:FOMC 决议日 14:00 公告窗口与 14:30 发布会窗口的检出率 vs 非 FOMC 日同窗口的基础率。
自检:未来数据投毒(截断后 bar ×7.3,此前的标记必须逐位不变)。

用法: python cojump_cos_feasibility.py [--pairs QQQ:SPY,QQQ:IWM,...] [--phases ...]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _P  # noqa: F401
from etft.intraday.bars import load_bars
import cojump_cos as CJ
from event_calendar import FOMC_DECISION

TAB = Path("../outputs/tables")
TAB.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261020)


def aligned_panels(sa: str, sb: str):
    A, B = load_bars(sa), load_bars(sb)
    days = A.days.intersection(B.days)
    cols = [c for c in A.c.columns if c in B.c.columns]
    ok = lambda X: X.loc[days, cols]
    m = ok(A.c).notna() & ok(B.c).notna() & ok(A.o).notna() & ok(B.o).notna()
    # 只保留两资产都有的 bar;缺口较多的日子整体丢弃(<95% 的 bar)
    keep = m.mean(axis=1) >= 0.95
    m = m.loc[keep]
    def pan(X):
        class _B: pass
        o = _B()
        for k in "ohlc":
            setattr(o, k, getattr(X, k).loc[m.index, cols])
        return o
    return CJ.long_panel(pan(A), m), CJ.long_panel(pan(B), m), m


def shifted_B(dfB: pd.DataFrame, shift_days: int = 1) -> pd.DataFrame:
    """把 B 的内容整体后移 shift_days 个交易日(行对行),day/clock 保持 A 的。"""
    out = dfB.copy()
    days = out["day"].unique()
    d2i = {d: i for i, d in enumerate(days)}
    idx = out["day"].map(d2i).to_numpy()
    g = out.groupby("day", sort=False).cumcount().to_numpy()
    # 以 (日序号-shift, 日内序) 取 B 的值;缺失则用 NaN 并在检测中被丢弃
    key = pd.Series(np.arange(len(out)), index=pd.MultiIndex.from_arrays([idx, g]))
    src = key.reindex(pd.MultiIndex.from_arrays([idx - shift_days, g])).to_numpy()
    ok = ~np.isnan(src)
    res = out.copy()
    for k in "ohlc":
        v = np.full(len(out), np.nan); v[ok] = out[k].to_numpy()[src[ok].astype(int)]
        res[k] = v
    return res


def run_detect(dfA, dfB, alpha):
    d = CJ.detect(dfA, dfB, alpha)
    return d[d["T"].notna()]


def day_rate(det, col):
    g = det.groupby("day")[col].any()
    return float(g.mean()), int(det[col].sum()), len(g)


def inject(df: pd.DataFrame, rows: np.ndarray, amp: float, sign: np.ndarray, sig: np.ndarray):
    """把指定行替换成光头光脚的 K 线:收盘 = 开盘·exp(sign·amp·sig),无影线。"""
    out = df.copy()
    o = out["o"].to_numpy()[rows]
    c = o * np.exp(sign * amp * sig)
    out.loc[out.index[rows], "c"] = c
    out.loc[out.index[rows], "h"] = np.maximum(o, c)
    out.loc[out.index[rows], "l"] = np.minimum(o, c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="QQQ:SPY,QQQ:IWM,QQQ:XLK,QQQ:TLT")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--ninj", type=int, default=150)
    a = ap.parse_args()
    fomc = set(pd.to_datetime(FOMC_DECISION))
    rows_c1, rows_c2, rows_c3, notes = [], [], [], []
    for pair in a.pairs.split(","):
        sa, sb = pair.split(":")
        dfA, dfB, m = aligned_panels(sa, sb)
        print(f"\n===== {sa}-{sb}: {dfA['day'].nunique()} 天, {len(dfA)} 根 bar =====")
        # ---------- 自检:未来投毒
        cut = dfA["day"].unique()[int(0.6 * dfA["day"].nunique())]
        det0 = run_detect(dfA, dfB, a.alpha)
        pA, pB = dfA.copy(), dfB.copy()
        mk = pA["day"] > cut
        for k in "ohlc":
            pA.loc[mk, k] *= 7.3; pB.loc[mk, k] *= 7.3
        det1 = run_detect(pA, pB, a.alpha)
        both = det0.merge(det1, on=["day", "clock"], suffixes=("_0", "_1"))
        both = both[both["day"] <= cut]
        same = bool((both["cj_nom_0"] == both["cj_nom_1"]).all() and (both["cj_fit_0"] == both["cj_fit_1"]).all()
                    and np.allclose(both["T_0"], both["T_1"], equal_nan=True))
        print(f"  [{'PASS' if same else 'FAIL'}] 未来投毒:cut={str(cut)[:10]} 之前 {len(both)} 根 bar 的 T 与标记逐位一致")
        notes.append((pair, "poison", same))
        # ---------- C1 校准
        detP = run_detect(dfA, shifted_B(dfB, 1), a.alpha)
        for col in ("cj_nom", "cj_fit"):
            r_d, r_n, nd = day_rate(det0, col)
            p_d, p_n, npd = day_rate(detP, col)
            cand_rate = float(det0["cand"].mean())
            print(f"  C1 {col}: 真实 {r_n} 个标记/{r_d:.1%} 的日子有标记 | 错开一天的零分布 {p_n} 个/{p_d:.1%} | 候选占比 {cand_rate:.1%}")
            rows_c1.append(dict(pair=pair, rule=col, real_flags=r_n, real_day_rate=r_d, null_flags=p_n, null_day_rate=p_d,
                                ratio=(r_n / max(p_n, 1)), cand_rate=cand_rate, n_days=nd))
        # ---------- C3 事件命中(FOMC 14:00 公告窗口、14:30 发布会窗口)
        def win(lo, hi):
            return (det0["clock"] >= lo) & (det0["clock"] <= hi)
        for name, lo, hi in (("FOMC_14:00-14:05", "14:01", "14:05"), ("FOMC_14:30-14:45", "14:31", "14:45")):
            isf = det0["day"].isin(fomc)
            w_ = win(lo, hi)
            for col in ("cj_nom", "cj_fit"):
                f = det0[isf & w_].groupby("day")[col].any()
                n = det0[~isf & w_].groupby("day")[col].any()
                rows_c3.append(dict(pair=pair, window=name, rule=col, fomc_days=len(f), fomc_hit=float(f.mean()),
                                    nonfomc_days=len(n), base_rate=float(n.mean())))
                print(f"  C3 {name} {col}: FOMC日命中 {f.mean():.1%} ({int(f.sum())}/{len(f)}) vs 非FOMC日同窗口 {n.mean():.1%}")
        # ---------- C2 功效/特异性:注入
        sig = (dfA["o"].pipe(lambda s: np.log(dfA["c"] / dfA["o"])).abs()).rolling(CJ.W, min_periods=CJ.W).median().shift(1) / 0.6745
        sigB = (np.log(dfB["c"] / dfB["o"]).abs()).rolling(CJ.W, min_periods=CJ.W).median().shift(1) / 0.6745
        clock = dfA["clock"].to_numpy()
        elig = np.where((clock >= "10:00") & (clock <= "15:30") & sig.notna().to_numpy() & sigB.notna().to_numpy())[0]
        days_arr = dfA["day"].to_numpy()
        # 每个日子至多注入一次,且注入点之间相隔 > 3 个交易日,避免互相污染窗口
        pick, last_day_i = [], -10
        udays = {d: i for i, d in enumerate(pd.unique(days_arr))}
        cand_rows = RNG.permutation(elig)
        used_days = []
        for r in cand_rows:
            di = udays[days_arr[r]]
            if all(abs(di - u) > 3 for u in used_days):
                pick.append(r); used_days.append(di)
            if len(pick) >= a.ninj:
                break
        pick = np.array(sorted(pick))
        sign = RNG.choice([-1.0, 1.0], size=len(pick))
        for amp in (3.0, 5.0, 8.0):
            for kind in ("co", "idio"):
                iA = inject(dfA, pick, amp, sign, sig.to_numpy()[pick])
                iB = inject(dfB, pick, amp, sign, sigB.to_numpy()[pick]) if kind == "co" else dfB
                det = run_detect(iA, iB, a.alpha).reset_index(drop=True)
                key = det.set_index(["day", "clock"])
                hit = {}
                for col in ("cj_nom", "cj_fit"):
                    h = [bool(key[col].get((days_arr[r], clock[r]), False)) for r in pick]
                    hit[col] = float(np.mean(h))
                rows_c2.append(dict(pair=pair, amp_sigma=amp, kind=kind, n_inj=len(pick), power_nom=hit["cj_nom"], power_fit=hit["cj_fit"]))
                print(f"  C2 注入 {kind:4s} 幅度 {amp:.0f}σ: 同 bar 检出率 nom {hit['cj_nom']:.1%}  fit {hit['cj_fit']:.1%}  (n={len(pick)})")
    pd.DataFrame(rows_c1).to_csv(TAB / "cojump_cos_C1_calibration.csv", index=False)
    pd.DataFrame(rows_c2).to_csv(TAB / "cojump_cos_C2_injection.csv", index=False)
    pd.DataFrame(rows_c3).to_csv(TAB / "cojump_cos_C3_fomc.csv", index=False)
    print("\n自检汇总:", {p: ok for p, k, ok in notes})
    if not all(ok for _, _, ok in notes):
        sys.exit(1)


if __name__ == "__main__":
    main()
