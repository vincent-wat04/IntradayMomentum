"""B0 验证:对 b0_econ 里看起来最好的 DUR 格子做稳健性 + 家族级置换安慰剂。

纪律(写在跑之前):
  * 候选格子 = 在 b0_econ 中预先"看过"的 70 个格子(7 特征集 × 2 模型 × {skip10/20/30, cont0.5/1.0}),全部计入多重性。
  * 置换:同一个"年内逐笔置换"同时作用于 14 条预测向量(保留格子间相关),重算因果分位与全部 70 格,取
      T1 = max_cells 全OOS Sleeve ΔSR,   T2 = max_cells min(dev-OOS, test) Sleeve ΔSR。
    观测值的 family-wise p = P(T_perm >= T_obs)。
  * 置换只破坏"预测—笔盈亏"的对应,保留笔盈亏的年度/相位结构 → 回答"随便一条预测向量在 70 格里挑最大能到多少"。
  * 稳健性(非选择):邻近 q、去掉贡献最大 5 个交易日、多空拆分、分年、分布面板。
"""
from __future__ import annotations

import bisect
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b0_entry_features as B0
from b0_econ import causal_rank, panel, sharpe

DATA, TAB = B0.DATA, B0.TAB
NPH = 30
PER = {"dev": ("2021-01-01", "2022-12-31"), "test": ("2023-01-01", "2026-12-31"), "full": ("2021-01-01", "2026-12-31")}
RULES = [("skip", 0.10), ("skip", 0.20), ("skip", 0.30), ("cont", 0.5), ("cont", 1.0)]


class Ctx:
    def __init__(self):
        b = B0.load_bars("QQQ"); self.days = b.days
        TR = pd.read_pickle(DATA / "_TR.pkl").sort_values(["day", "phase", "k_dec"]).reset_index(drop=True)
        self.TR = TR; self.day_arr = TR.day.to_numpy(); self.pnl = TR.pnl.to_numpy()
        self.day_idx = pd.Index(self.days).get_indexer(TR.day)             # 每笔对应交易日序号
        self.nd = len(self.days)
        self.preds = pd.read_pickle(DATA / "_B0_preds.pkl")
        self.side = TR.side.to_numpy(); self.year = TR.day.dt.year.to_numpy()
        self.per_mask = {k: (self.days >= a) & (self.days <= b_) for k, (a, b_) in PER.items()}
        import trend_rv_sleeve_study as T, round3_common as C
        from sens_rv_weights import pct_w
        px = C.load_close("yahoo/QQQ.parquet"); rr = px.pct_change()
        w = (0.5 + 1.0 * (1 - pct_w(rr.rolling(21).std(), 504))).shift(1).reindex(self.days)
        self.L1 = (w * (b.day_close / b.prev_close_adj - 1).reindex(self.days)).to_numpy()
        self.base = self.curve(np.ones(len(TR), bool))

    def curve(self, keep, scale=None, mask=None):
        v = self.pnl * (1.0 if scale is None else scale)
        v = np.where(keep, v, 0.0)
        if mask is not None: v = np.where(mask, v, 0.0)
        s = np.bincount(self.day_idx, weights=v, minlength=self.nd) / NPH
        return s

    def sr(self, x, per):
        m = self.per_mask[per]; xs = x[m]
        return xs.mean() / xs.std() * np.sqrt(252) if xs.std() > 0 else np.nan

    def rule_curve(self, pr, rank, rule, q):
        fin = np.isfinite(pr)
        if rule == "skip":
            return self.curve(~(fin & (rank <= q)))
        sc = np.clip(np.exp(q * 2 * (rank - 0.5)), 0.5, 1.5)
        return self.curve(np.ones(len(pr), bool), scale=sc)

    def dsr(self, c, per, port=False):
        if port: return self.sr(self.L1 + c, per) - self.sr(self.L1 + self.base, per)
        return self.sr(c, per) - self.sr(self.base, per)


def all_cells(ctx, preds):
    out = {}
    for key, pr in preds.items():
        rk = causal_rank(pr, ctx.day_arr)
        for rule, q in RULES:
            c = ctx.rule_curve(pr, rk, rule, q)
            out[(key, rule, q)] = (ctx.dsr(c, "dev"), ctx.dsr(c, "test"), ctx.dsr(c, "full"))
    return out


def permute_within_year(preds, ctx, rng):
    perm = np.arange(len(ctx.TR))
    for y in np.unique(ctx.year):
        ix = np.where(ctx.year == y)[0]; perm[ix] = rng.permutation(ix)
    return {k: v[perm] for k, v in preds.items()}


def main():
    t0 = time.time(); ctx = Ctx(); preds = ctx.preds
    obs = all_cells(ctx, preds)
    T1o = max(v[2] for v in obs.values()); T2o = max(min(v[0], v[1]) for v in obs.values())
    best1 = max(obs.items(), key=lambda kv: kv[1][2]); best2 = max(obs.items(), key=lambda kv: min(kv[1][0], kv[1][1]))
    print(f"[ctx+obs] {time.time()-t0:.1f}s  观测 T1(max 全OOS ΔSR)={T1o:.3f} @ {best1[0]};  T2(max min(dev,test))={T2o:.3f} @ {best2[0]}")

    # ---------- 1. 稳健性(针对 3 条线:F1+DUR lgbm / F1+DUR ridge / F1 lgbm 基线) ----------
    lines = [(("F1+DUR", "lgbm")), (("F1+DUR", "ridge")), (("F1", "lgbm"))]
    rows = []
    for key in lines:
        pr = preds[key]; rk = causal_rank(pr, ctx.day_arr); fin = np.isfinite(pr)
        for q in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
            c = ctx.rule_curve(pr, rk, "skip", q)
            rows.append(dict(line="|".join(key), test="neighbor_q", q=q, dev=ctx.dsr(c, "dev"), test_=ctx.dsr(c, "test"), full=ctx.dsr(c, "full"), port_full=ctx.dsr(c, "full", True)))
        keep = ~(fin & (rk <= 0.20)); c = ctx.rule_curve(pr, rk, "skip", 0.20)
        # 去掉贡献最大的 5 个交易日(按 c - base 的日差额)
        diff = c - ctx.base; m = ctx.per_mask["full"]
        idx = np.where(m)[0]; top = idx[np.argsort(-diff[idx])[:5]]
        c2 = c.copy(); b2 = ctx.base.copy(); c2[top] = 0; b2[top] = 0     # 两边同时去掉这 5 日
        f = lambda x: x[m].mean() / x[m].std() * np.sqrt(252)
        rows.append(dict(line="|".join(key), test="drop_top5_days", q=0.20, dev=np.nan, test_=np.nan, full=f(c2) - f(b2), port_full=np.nan))
        # 多空拆分:只对一侧应用跳过
        for sd, nm in ((1, "long_only_skip"), (-1, "short_only_skip")):
            kp = ~(fin & (rk <= 0.20) & (ctx.side == sd)); c3 = ctx.curve(kp)
            rows.append(dict(line="|".join(key), test=nm, q=0.20, dev=ctx.dsr(c3, "dev"), test_=ctx.dsr(c3, "test"), full=ctx.dsr(c3, "full"), port_full=ctx.dsr(c3, "full", True)))
        # 分年
        for y in range(2021, 2027):
            ym = (pd.Series(ctx.days).dt.year.to_numpy() == y)
            s1 = c[ym]; s0 = ctx.base[ym]
            if s1.std() > 0: rows.append(dict(line="|".join(key), test=f"year_{y}", q=0.20, dev=np.nan, test_=np.nan, full=s1.mean() / s1.std() * np.sqrt(252) - s0.mean() / s0.std() * np.sqrt(252), port_full=np.nan))
        # 跳过的笔 vs 保留的笔平均盈亏(bp,按 y 列不可得,用 pnl)
        sk = fin & (rk <= 0.20)
        rows.append(dict(line="|".join(key), test="skipped_vs_kept_pnl_bp", q=0.20, dev=ctx.pnl[sk & (ctx.year <= 2022)].mean() * 1e4, test_=ctx.pnl[sk & (ctx.year >= 2023)].mean() * 1e4, full=ctx.pnl[fin & ~sk].mean() * 1e4, port_full=sk.sum()))
    R = pd.DataFrame(rows); pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print("\n== 稳健性(Sleeve ΔSR vs 不过滤;'skipped_vs_kept' 行:dev=跳过笔均盈亏(bp,dev),test_=跳过笔(test),full=保留笔均盈亏,port_full=跳过笔数) ==")
    print(R.round(3).to_string(index=False)); R.to_csv(TAB / "b0_validate_robust.csv", index=False)

    # ---------- 2. 分布面板(F1+DUR lgbm skip20% vs F1 lgbm skip20% vs 基线;Sleeve 层与组合层) ----------
    pan = {}
    m = ctx.per_mask["full"]
    for nm, key in (("F1+DUR lgbm skip20", ("F1+DUR", "lgbm")), ("F1+DUR ridge skip20", ("F1+DUR", "ridge")), ("F1 lgbm skip20", ("F1", "lgbm"))):
        pr = preds[key]; rk = causal_rank(pr, ctx.day_arr); c = ctx.rule_curve(pr, rk, "skip", 0.20)
        pan["Sleeve " + nm] = panel(pd.Series(c[m])); pan["组合 " + nm] = panel(pd.Series((ctx.L1 + c)[m]))
    pan["Sleeve 基线"] = panel(pd.Series(ctx.base[m])); pan["组合 基线"] = panel(pd.Series((ctx.L1 + ctx.base)[m]))
    P = pd.DataFrame(pan).T; print("\n== 分布面板(2021–2026 OOS) =="); print(P.round(3).to_string()); P.to_csv(TAB / "b0_validate_panel.csv")

    # ---------- 3. 家族级置换 ----------
    NP = int(os.environ.get("NPERM", "200")); rng = np.random.default_rng(20261020)
    t1 = np.empty(NP); t2 = np.empty(NP); t0 = time.time()
    for i in range(NP):
        pp = permute_within_year(preds, ctx, rng); cells = all_cells(ctx, pp)
        t1[i] = max(v[2] for v in cells.values()); t2[i] = max(min(v[0], v[1]) for v in cells.values())
        if i == 2: print(f"  [perm] 每次 {(time.time()-t0)/3:.1f}s,预计 {(time.time()-t0)/3*NP/60:.1f} 分钟", flush=True)
    p1 = (np.sum(t1 >= T1o) + 1) / (NP + 1); p2 = (np.sum(t2 >= T2o) + 1) / (NP + 1)
    print(f"\n== 家族级置换(NP={NP}, 年内逐笔置换, 同一置换作用于全部 14 条预测) ==")
    print(f"T1 max 全OOS ΔSR: 观测 {T1o:.3f};置换分布 median {np.median(t1):.3f}, 95% {np.quantile(t1,.95):.3f}, 99% {np.quantile(t1,.99):.3f};  p_fw={p1:.4f}")
    print(f"T2 max min(dev,test) ΔSR: 观测 {T2o:.3f};置换分布 median {np.median(t2):.3f}, 95% {np.quantile(t2,.95):.3f}, 99% {np.quantile(t2,.99):.3f};  p_fw={p2:.4f}")
    # 单格(F1+DUR lgbm skip20)的非家族 p
    k0 = (("F1+DUR", "lgbm"), "skip", 0.20); print(f"单格 {k0}: 观测 全OOS {obs[k0][2]:.3f}, dev {obs[k0][0]:.3f}, test {obs[k0][1]:.3f}")
    pd.DataFrame({"T1": t1, "T2": t2}).to_csv(TAB / "b0_validate_perm.csv", index=False)
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
