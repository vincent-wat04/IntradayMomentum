"""让仓位模型自己学 FFA 跳过(2026-10-23,事后;新配置 +5)。
训练样本包含 FFA 跳过的低振幅日;给模型跳过规则本身的指示变量 ffa_skip = 1[ffa_rank ≤ 0.2](阈值形状,线性/sigmoid 能表达)。
  G1 训练含跳过日 + 指示变量,交易时不硬跳过(模型自己决定)
  G2 训练含跳过日 + 指示变量,交易时硬跳过
  G3 训练含跳过日、无指示变量,交易时硬跳过(只看训练样本变化的影响)
底座 v4(E1 入场带 + CLEAN+dur_rate + 线性 + 均值–方差);另用 v3(sigmoid、原入场带)跑 G1/G2 作交叉检验。
报告:相对 v4 / v3 的组合 ΔSR、分年;G1 在跳过日学到的 k;指示变量逐折系数(换算为对 k 的直接影响)。
python3 r3_ffa_learn.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_pred_study as DP
import liquidity_inverse_study as LI
import r2_rules_study as RR
import r3_study as R3
import sizing_g5_followup as G
import sizing_models_study as M
import sizing_sigma_feed as SF
import sota_unified_stack as SU
import unified_sizing_posthoc as P
from etft.intraday.bars import load_bars


def main():
    t0 = time.time(); pd.set_option("display.width", 280, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); keepF = I["keep"]; keep1 = pd.Series(1.0, index=days)
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]; X["ffa_skip"] = (X.ffa_rank <= 0.2).astype(float)
    assert ((X.ffa_skip == 1) == (keepF == 0))[sel].all()
    tr_share = {yr: float(X.ffa_skip[sel & (yrs < yr)].mean()) for yr in range(2021, 2027)}; print("[训练样本中跳过日占比]", {k: round(v, 3) for k, v in tr_share.items()})
    _, S3m, _ = RR.band_multipliers(b)
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b); PHbE, PHfE = R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True)
    LG = {("base", "F"): R3.legs_k(PHb, PHf, b, keepF), ("base", "N"): R3.legs_k(PHb, PHf, b, keep1),
          ("E1", "F"): R3.legs_k(PHbE, PHfE, b, keepF), ("E1", "N"): R3.legs_k(PHbE, PHfE, b, keep1)}
    F4, F3 = P.CLEAN + ["dur_rate"], P.CLEAN
    R, K = {}, {}
    def run(nm, lg, feats, lk, train, trade):
        k = R3.sizing(X, feats, LG[(lg, train)], I, sel, days, "mv", lk); K[nm] = (k, lg, feats, lk, train)
        R[nm] = R3.assemble(k, LG[(lg, trade)], I, om)
    run("v4(训练不含跳过日,硬跳过)", "E1", F4, "lin", "F", "F")
    run("G1 v4:训练含跳过日+指示变量,不硬跳过", "E1", F4 + ["ffa_skip"], "lin", "N", "N")
    run("G2 v4:训练含跳过日+指示变量,硬跳过", "E1", F4 + ["ffa_skip"], "lin", "N", "F")
    run("G3 v4:训练含跳过日、无指示变量,硬跳过", "E1", F4, "lin", "N", "F")
    run("v4 去 FFA(参照:训练含跳过日、不硬跳过)", "E1", F4, "lin", "N", "N")
    run("v3(参照)", "base", F3, "sig", "F", "F")
    run("G1 v3:训练含跳过日+指示变量,不硬跳过", "base", F3 + ["ffa_skip"], "sig", "N", "N")
    run("G2 v3:训练含跳过日+指示变量,硬跳过", "base", F3 + ["ffa_skip"], "sig", "N", "F")
    print(f"[完成] ({time.time()-t0:.0f}s)", flush=True)
    o = om.to_numpy(); oi = np.where(o)[0]; sk = (keepF == 0).to_numpy() & o; ns = (keepF == 1).to_numpy() & o
    dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    rows = []
    for nm, r_ in R.items():
        refn = "v3(参照)" if "v3" in nm else "v4(训练不含跳过日,硬跳过)"; ref = R[refn]["port"].to_numpy()[oi]; x_ = r_["port"].to_numpy()[oi]
        db = np.array([srv(x_[q]) - srv(ref[q]) for q in BI]); k = r_["k"]
        rows.append(dict(配置=nm, 参照=refn.split("(")[0], 组合SR_OOS=srv(x_), ΔSR=srv(x_) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                         dev=srv(x_[dev]) - srv(ref[dev]), test=srv(x_[~dev]) - srv(ref[~dev]),
                         跳过日_k多=float(k.kl[sk].mean()), 跳过日_k空=float(k.ks[sk].mean()), 其余日_k多=float(k.kl[ns].mean()), 其余日_k空=float(k.ks[ns].mean()),
                         跳过日_k为0比例=float(np.mean(np.r_[k.kl[sk].to_numpy(), k.ks[sk].to_numpy()] <= 1e-6)),
                         **{str(y): srv(x_[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)}))
    T = pd.DataFrame(rows); print("\n=== 组合层 OOS 2021–26(ΔSR 相对各自参照)==="); print(T.to_string(index=False))
    # ---- 指示变量的逐折系数(对 k 的直接影响 = θ / 训练期标准差;线性链接下即 k 的变化量,sigmoid 下为 logit 尺度的 θ/sd,换算 Δk ≈ θ/sd·k(1−k/2))
    rows = []
    for nm in ("G1 v4:训练含跳过日+指示变量,不硬跳过", "G1 v3:训练含跳过日+指示变量,不硬跳过"):
        k, lg, feats, lk, train = K[nm]; L = LG[(lg, train)]
        Lb, Sb, Eb = (L["L2"] * 1e4).to_numpy(), (L["S2"] * 1e4).to_numpy(), ((I["leg1"] + I["ov"]) * 1e4).to_numpy()
        k2, folds, lam = G.wf_theta(X, feats, Lb, Sb, Eb, sel, days, lk)
        assert float(np.nanmax(np.abs(k2.to_numpy() - k.to_numpy()))) < 1e-6, "θ 重算与 k 不一致"
        j = feats.index("ffa_skip") + 1
        for yr, (th, mu, sd) in folds.items():
            s_ = float(sd["ffa_skip"]); rows.append(dict(模型=nm.split(":")[0], 折=yr, λ=lam[yr], θ多_std=th[0, j], θ空_std=th[1, j], 指示变量对k多的影响=th[0, j] / s_, 指示变量对k空的影响=th[1, j] / s_))
    TC = pd.DataFrame(rows); print("\n=== 指示变量 ffa_skip 的逐折系数(线性:直接是 Δk;sigmoid:logit 尺度)==="); print(TC.to_string(index=False))
    # ---- 跳过日的损益:G1 是否把它们变成了赚钱日
    rows = []
    for nm in ("G1 v4:训练含跳过日+指示变量,不硬跳过", "v4 去 FFA(参照:训练含跳过日、不硬跳过)", "G1 v3:训练含跳过日+指示变量,不硬跳过"):
        s_ = R[nm]["sleeve"].to_numpy()
        rows.append(dict(配置=nm, **{str(y): s_[sk & (yrs == y)].mean() * 1e4 if (sk & (yrs == y)).any() else np.nan for y in range(2021, 2027)}, 全部=s_[sk].mean() * 1e4))
    print("\n=== 跳过日上的 Sleeve 日均损益(bp,仓位缩放后)==="); print(pd.DataFrame(rows).to_string(index=False))
    T.to_csv(M.TAB / "r3_ffa_learn.csv", index=False); TC.to_csv(M.TAB / "r3_ffa_learn_coef.csv", index=False); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
