"""Where does the v5 core weight go? Decompose v4 -> v5 and compare with the two-asset mean-variance allocation.
python3 audit/explain_v5.py   (from studies/)  -> outputs/tables/explain_v5.csv
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import review_checks as RC
import noise_core_dd as NC, tearsheet_v5 as T5, round3_common as C

TAB = HERE.parent.parent / "outputs" / "tables"


def main():
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:9.4f}")
    E = RC.setup(); base = RC.v4(E); days, om, I = E["days"], E["om"], E["I"]; o = om.to_numpy(); n = len(days)
    ev = I["e_vt"].fillna(0).to_numpy(); bh = I["bh"].to_numpy(); s4 = base["sleeve"].to_numpy(); p4 = base["port"].to_numpy()
    kmax = np.maximum(base["k"].kl.where(om, 1.0).fillna(1.0), base["k"].ks.where(om, 1.0).fillna(1.0)).to_numpy()
    w = NC.core_weights(C.load_close("yahoo/QQQ.parquet"))[0]["base"].reindex(I["bh"].index).ffill().fillna(1.0).reindex(days).fillna(1.0).to_numpy()
    core4 = w * bh
    print("=== 1. Who sizes what: the PPP multipliers apply to the Sleeve legs only; the core weight is the fixed rule w_t ===")
    print(f"core rule weight w_t (OOS): mean {w[o].mean():.3f}, min {w[o].min():.2f}, max {w[o].max():.2f};  PPP k_long mean {base['k'].kl[om].mean():.3f}, k_short mean {base['k'].ks[om].mean():.3f}")
    steps = {"v4": None, "+cap": dict(use_ratio=False, use_band=False), "+cap+band": dict(use_ratio=False), "v5 (rho=1.5)": dict()}
    rows = []
    for nm, kw in steps.items():
        if kw is None:
            r = dict(core_w=w, sleeve_mult=np.ones(n), f=np.ones(n), c=np.ones(n), port=p4)
        else:
            r = T5.build_v5(w, s4, ev, kmax, bh, p4, **kw)
        x = r["port"]; sr = lambda a: a.mean() / a.std() * np.sqrt(252)
        rows.append(dict(config=nm, core_w_mean=r["core_w"][o].mean(), sleeve_mult_mean=r["sleeve_mult"][o].mean(), c_mean=r["c"][o].mean(), f_mean=r["f"][o].mean(),
                         share_days_f_lt_1=(r["f"][o] < 1 - 1e-9).mean(), core_vol_ann=(r["core_w"] * bh)[o].std() * np.sqrt(252), sleeve_vol_ann=(r["sleeve_mult"] * s4)[o].std() * np.sqrt(252),
                         port_vol_ann=x[o].std() * np.sqrt(252), oos_sharpe=sr(x[o])))
    D = pd.DataFrame(rows); print(D.to_string(index=False)); D.to_csv(TAB / "explain_v5.csv", index=False)
    print("\n=== 2. Two-asset mean-variance view (OOS 2021-26, v4 legs, before any rescaling) ===")
    mu = np.array([core4[o].mean(), s4[o].mean()]) * 252; S = np.cov(np.vstack([core4[o], s4[o]])) * 252; corr = S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])
    sr_c, sr_s = mu[0] / np.sqrt(S[0, 0]), mu[1] / np.sqrt(S[1, 1])
    wmv = np.linalg.solve(S, mu); rho_mv = wmv[1] / wmv[0]
    print(f"core: mean {mu[0]:.3f}/yr, vol {np.sqrt(S[0,0]):.3f}, Sharpe {sr_c:.2f};  Sleeve(v4, mult 1): mean {mu[1]:.3f}/yr, vol {np.sqrt(S[1,1]):.3f}, Sharpe {sr_s:.2f};  corr {corr:+.2f}")
    print(f"unconstrained max-Sharpe weights (relative to v4's 1:1 mix): sleeve/core = {rho_mv:.2f}  (v5 uses rho = 1.5 then rescales total risk back to v4's level)")
    print(f"risk share of Sleeve in v4: {S[1,1]/ (S[0,0]+S[1,1]+2*S[0,1]):.2f} of variance (own variance / total), core: {S[0,0]/(S[0,0]+S[1,1]+2*S[0,1]):.2f}")
    print("\n=== 3. Does the cap/ratio act mostly on days when the Sleeve is on? ===")
    r5 = T5.build_v5(w, s4, ev, kmax, bh, p4); active = (np.abs(s4) > 0) & o
    print(f"core weight reduction vs rule (w - executed): active-Sleeve days {np.mean(w[active]-r5['core_w'][active]):.3f}, other days {np.mean(w[o & ~active]-r5['core_w'][o & ~active]):.3f}")


if __name__ == "__main__":
    main()
