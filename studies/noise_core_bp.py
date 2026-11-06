"""买力约束下的配比(事后,2026-10-29;新配置 +2)。
实盘最坏情形占用 ≈ 核心仓权重 w + Sleeve 敞口 e·k/3(TQQQ 实现 1/3 名义),单位 CAPITAL_BASE;账户买力 ≈ 3.03 × CAPITAL_BASE
(preflight:买力 $1,513,400 / CAPITAL_BASE $500,000),护栏 85% → 上限 2.57。
对 v4、S1、S2 统一施加:当日 w + e·max(k)·m/3 > 2.57 时,两条腿按比例缩到上限(用当日开盘前可得的量,PIT)。
python3 noise_core_bp.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crowding_daily_strategy as CS
import disp_pred_study as DP
import noise_core_dd as N
import r2_rules_study as RR
import r3_study as R3
import round3_common as C
import sizing_models_study as M
import sizing_sigma_feed as SF
import unified_sizing_posthoc as P
from etft.intraday.bars import load_bars

CAP = 0.85 * 1_513_400 / 500_000


def main():
    pd.set_option("display.width", 260, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); keep = I["keep"]; bh = I["bh"].to_numpy(); ev = I["e_vt"].fillna(0).to_numpy()
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days); oo = om.to_numpy()
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]
    _, S3m, _ = RR.band_multipliers(b); Lv4 = R3.legs_k(R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True), b, keep)
    v4 = R3.assemble(R3.sizing(X, P.CLEAN + ["dur_rate"], Lv4, I, sel, days, "mv", "lin"), Lv4, I, om)
    sl4 = v4["sleeve"].to_numpy(); kmax = np.maximum(v4["k"].kl.where(om, 1.0).fillna(1.0), v4["k"].ks.where(om, 1.0).fillna(1.0)).to_numpy()
    px = C.load_close("yahoo/QQQ.parquet"); W, _ = N.core_weights(px); wb = W["base"].reindex(I["bh"].index).ffill().fillna(1.0).reindex(days).fillna(1.0).to_numpy()
    legd = lambda w: w * bh - 2e-4 * np.abs(np.diff(np.r_[w[0], w]))
    srv = lambda v: v.mean() / v.std() * np.sqrt(252); oi = np.where(oo)[0]; dev = yrs[oi] <= 2022
    def capped(cw, sm):
        """cw: 核心仓乘数数组;sm: Sleeve 乘数数组。返回组合收益、最坏占用(缩放前)、缩放因子。"""
        use = cw * wb + ev * kmax * sm / 3.0; f = np.minimum(1.0, CAP / np.maximum(use, 1e-9)); f = np.where(oo, f, 1.0)
        w = cw * wb * f; return legd(w) + f * sm * sl4, use, f
    one = np.ones(len(days))
    def mk_S1():
        cands = {m: capped(one, np.where(oo, m, 1.0))[0] for m in N.MS}; return CS.recursive(cands, days, sel)
    def mk_S2():
        base = pd.Series(capped(one, one)[0], index=days); cands = {}
        for rho in sorted(N.RHOS, reverse=True):
            mix = pd.Series(legd(wb) + rho * sl4, index=days)
            c_ = (base.rolling(252, min_periods=126).std() / mix.rolling(252, min_periods=126).std()).shift(1).clip(0.5, 1.5).fillna(1.0).to_numpy()
            c_ = np.where(oo, c_, 1.0); cands[rho] = capped(c_, np.where(oo, c_ * rho, 1.0))[0]
        return CS.recursive(cands, days, sel)
    rows = []
    v4u, use0, _ = legd(wb) + sl4, wb + ev * kmax / 3, None
    v4c, use, f = capped(one, one)
    for nm, x, u in (("v4(不设上限)", v4u, use0), ("v4c(上限 2.57)", v4c, use)):
        rows.append(dict(方案=nm, 组合SR=srv(x[oi]), dev=srv(x[oi][dev]), test=srv(x[oi][~dev]), MDD=N.stats(x[oi])["MDD"], 最坏占用_最大=u[oo].max(), 最坏占用_p99=np.quantile(u[oo], .99), 超限天数比例=float((u[oo] > CAP).mean())))
    s1, p1 = mk_S1(); s2, p2 = mk_S2()
    for nm, x, pk in (("S1c Sleeve × m(上限 2.57)", s1, p1), ("S2c 等风险换配比(上限 2.57)", s2, p2)):
        rows.append(dict(方案=nm + f" 选择 {pd.Series([str(v) for v in pk.values()]).value_counts().to_dict()}", 组合SR=srv(x[oi]), dev=srv(x[oi][dev]), test=srv(x[oi][~dev]), MDD=N.stats(x[oi])["MDD"]))
    T = pd.DataFrame(rows); print(f"=== 买力约束(最坏占用 ≤ {CAP:.2f} × CAPITAL_BASE)下的组合 OOS 2021–26 ==="); print(T.to_string(index=False))
    BI = M.block_idx(len(oi), 10, 2000, 0); ref = v4c[oi]; refb = np.array([srv(ref[q]) for q in BI])
    for nm, x in (("S1c", s1), ("S2c", s2)):
        xs = x[oi]; db = np.array([srv(xs[q]) for q in BI]) - refb
        print(f"  {nm} 相对 v4c:ΔSR {srv(xs)-srv(ref):+.3f} [{np.percentile(db,2.5):+.3f}, {np.percentile(db,97.5):+.3f}];dev {srv(xs[dev])-srv(ref[dev]):+.3f} / test {srv(xs[~dev])-srv(ref[~dev]):+.3f};分年 "
              + " ".join(f"{y}:{srv(xs[yrs[oi]==y])-srv(ref[yrs[oi]==y]):+.2f}" for y in range(2021, 2027)))


if __name__ == "__main__":
    main()
