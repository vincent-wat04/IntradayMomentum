"""核对 PR #2(noise-band-validation)指出的两个 v5 记账问题,分别量化其影响(预注册,运行前写定):
 A  15:40 提前平仓没有计入主动平仓的半价差:G_series 把 MOC 出场价换成 15:41 开盘价,但成本仍按 MOC(只收佣金)。
    修正:对 15:40 仍持仓的每笔成交加 |shares|×$0.005(半价差),先从腿收益里扣掉,再重训 v4 仓位。
 B  v5 的买力缩放 f_t 用了 10:00 之后才有的 kmax,却乘到整日核心仓收益(含隔夜和 9:30–10:00)——候选相关的前视。
    修正:f_t 改用开盘前已知的上界 k = 2(PR #2 的做法),其余不变。
判据:ΔSR(相对原 v5,OOS 2021–26)的块自助 95% 区间;|ΔSR| < 0.05 视为"不改变结论",否则记入 REVIEW 并修正 tearsheet_v5。
python3 audit/pr2_accounting_check.py   (在 studies/ 下运行)
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
import r3_study as R3, noise_core_dd as NC, tearsheet_v5 as T5, round3_common as C, sota_unified_stack as SU, liquidity_inverse_study as LI
import ivol_objective_study as AO, unified_sizing_posthoc as P, sizing_sigma_feed as SF
from event_calendar import FOMC_DECISION

TAB = HERE.parent.parent / "outputs" / "tables"; HALF = 0.005


def exit_cost(PH, days, te="15:40"):
    TR = pd.concat([LI.trades_from_fills(F, off) for off, (rr, F) in PH.items() if len(F)], ignore_index=True)
    TR = TR[(TR.t_in < te) & (TR.t_out > te)].copy(); TR["X"] = TR.shares.abs() * HALF
    return [(TR[TR.side == s].groupby("day").X.sum() / 30.0).reindex(days).fillna(0.0) for s in (1, -1)]


def main():
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:9.4f}")
    E = RC.setup(); days, om, b, I = E["days"], E["om"], E["b"], E["I"]; o = om.to_numpy(); n = len(days)
    fm = pd.Series(days.isin(pd.DatetimeIndex(FOMC_DECISION)), index=days)
    XLb, XSb = exit_cost(E["PHb"], days); XLf, XSf = exit_cost(E["PHf"], days)
    XL, XS = XLb.where(~fm, XLf), XSb.where(~fm, XSf)
    legs0 = R3.legs_k(E["PHb"], E["PHf"], b, E["keep"])
    legs1 = dict(legs0); legs1["L2"] = legs0["L2"] - XL * E["keep"]; legs1["S2"] = legs0["S2"] - XS * E["keep"]
    print(f"[A] 15:40 仍持仓的主动出场半价差:多头腿 日均 {XL[o].mean()*1e4:.2f} bp,空头腿 {XS[o].mean()*1e4:.2f} bp(30 相位均值,占资金)")

    def v4_legs(legs):
        feats = P.CLEAN + ["dur_rate"]; X = E["X"]
        Lb, Sb, Eb = (legs["L2"] * 1e4).to_numpy(), (legs["S2"] * 1e4).to_numpy(), ((I["leg1"] + I["ov"]) * 1e4).to_numpy()
        Q = pd.DataFrame({"L": Lb, "S": Sb, "E": Eb}, index=days)
        k, _ = AO.wf_obj(X, feats, Lb, Sb, Eb, Q, E["sel"], days, "mv", "lin"); r = R3.assemble(k, legs, I, om); r["k"] = k; return r
    r0, r1 = v4_legs(legs0), v4_legs(legs1)
    ev = I["e_vt"].fillna(0).to_numpy(); bhv = I["bh"].to_numpy()
    w = NC.core_weights(C.load_close("yahoo/QQQ.parquet"))[0]["base"].reindex(I["bh"].index).ffill().fillna(1.0).reindex(days).fillna(1.0).to_numpy()
    kmax = lambda r: np.maximum(r["k"].kl.where(om, 1.0).fillna(1.0), r["k"].ks.where(om, 1.0).fillna(1.0)).to_numpy()
    two = np.full(n, 2.0)
    cfg = {"原 v5": (r0, kmax(r0)), "A 只修成本": (r1, kmax(r1)), "B 只修 PIT cap(k=2)": (r0, two), "A+B": (r1, two)}
    res = {}; ref = None
    for nm, (r, km) in cfg.items():
        v5 = T5.build_v5(w, r["sleeve"].to_numpy(), ev, km, bhv, r["port"].to_numpy(), rho=1.5); res[nm] = v5
    ref = res["原 v5"]["port"]; rows = []
    for nm, v5 in res.items():
        x = v5["port"]; d, lo, hi = RC.boot_delta(x, ref, o)
        te = o & (days.year >= 2023)
        rows.append(dict(配置=nm, OOS_SR=RC.srv(x[o]), test23_26_SR=RC.srv(x[te]), ΔSR=d, lo=lo, hi=hi, MDD=float((np.cumprod(1 + x[o]) / np.maximum.accumulate(np.cumprod(1 + x[o])) - 1).min()),
                         核心仓均权=float(v5["core_w"][o].mean()), 缩放f小于1天数比例=float((v5["f"][o] < 1 - 1e-9).mean()), 年化均值=float(x[o].mean() * 252)))
    R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(TAB / "pr2_accounting_check.csv", index=False)
    # v4 层面
    for nm, r in (("v4 原", r0), ("v4 修成本", r1)):
        print(f"{nm}: OOS SR {RC.srv(r['port'].to_numpy()[o]):.3f},Sleeve 单独 SR {RC.srv(r['sleeve'].to_numpy()[o]):.3f}")


if __name__ == "__main__":
    main()
