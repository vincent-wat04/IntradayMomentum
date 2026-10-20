"""B0 评估:新特征族相对 F1 的增量预测力(描述性,不选配置)。

1) 单变量 IC:每个特征 × {全部, 多, 空} × {dev-OOS 2021–22, test 2023–26},Spearman(特征, y),按交易日聚类自助 95% 区间;
   同时报"dev 与 test 同号且两段区间都不含 0"的特征。
2) 增量 IC:walk-forward(每年用此前所有年份训练),模型 = ridge / LightGBM,特征集:
     F1(基线)  vs  F1+DUR  F1+MIC  F1+BRP  F1+DUR+MIC+BRP  F1+VOLORTH(σ14 正交的波动族)
   指标:OOS IC、相对 F1 的 ΔIC 及自助区间(配对,按日聚类)。
3) 全程报告整张表,不挑最优;所有"好看"的结果都要通过"邻域/分方向/分段"稳定性检查才算线索。
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b0_entry_features as B0
import trade_filter_study as TF

DATA = B0.DATA; TAB = B0.TAB
F1 = TF.FEATS_F1
DUR = ["ev_rate", "ev_run", "ev_imb", "ev_same_frac", "ev_dur_last", "ev_dur_m", "ev_since"]
MIC = ["bvc_ofi5", "bvc_ofi15", "bvc_ofi30", "kyle15", "amihud15", "cs_spread15", "bo_vol_share", "turn_rel15"]
BRP = ["rv_near_share", "rv_near_ratio", "rv_late_w"]
STATE = ["side", "tod", "gap_dir", "log_sd14"]
VOLO = ["vol_vix_o", "vol_vxn_o", "vol_rvprev_o", "vol_b30_o"]               # 对 log σ14 正交后的波动族


def cluster_boot_ic(x, y, day_id, nb=600, seed=0):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, d = x[ok], y[ok], day_id[ok]
    if len(x) < 200:
        return np.nan, np.nan, np.nan, len(x)
    ic = spearmanr(x, y)[0]
    ud, inv = np.unique(d, return_inverse=True)
    order = np.argsort(inv, kind="stable"); st = np.searchsorted(inv[order], np.arange(len(ud))); en = np.append(st[1:], len(order))
    rg = np.random.default_rng(seed); bs = []
    for _ in range(nb):
        pick = rg.integers(0, len(ud), len(ud)); idx = np.concatenate([order[st[p]:en[p]] for p in pick]); bs.append(spearmanr(x[idx], y[idx])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5]); return ic, lo, hi, len(x)


def add_vol_orth(F, TR, Xd, yrs):
    """VIX/VXN/前日 RV/首 30 分钟 RV 对 log σ14 取残差。系数只用**训练年份**(每个 OOS 年用此前所有年份),此处先给出
    "用全部 2019–2020 估计"的固定版本用于单变量 IC;walk-forward 评估中按年重估。"""
    dpos = {d: i for i, d in enumerate(Xd.index)}; ii = TR.day.map(dpos).to_numpy()
    base = Xd.log_sd14.to_numpy()[ii]
    cols = {"vol_vix_o": Xd.log_vix.to_numpy()[ii], "vol_vxn_o": Xd.log_vxn.to_numpy()[ii], "vol_rvprev_o": Xd.log_rv_prev.to_numpy()[ii], "vol_b30_o": Xd.b_log_rv30.to_numpy()[ii]}
    return base, cols


def main():
    b = B0.load_bars("QQQ"); TR = pd.read_pickle(DATA / "_TR.pkl").sort_values(["day", "phase", "k_dec"]).reset_index(drop=True)
    ALL = pd.read_pickle(DATA / "_B0_features.pkl"); Xd, _, _, _ = B0.LV.build(b)
    yrs = TR.day.dt.year.to_numpy(); y = TR.y.to_numpy(); side = TR.side.to_numpy(); day_id = TR.day.dt.strftime("%Y-%m-%d").to_numpy()
    base, raw = add_vol_orth(ALL, TR, Xd, yrs)
    # 单变量用的正交化:用 2019–2020 估计回归系数(固定),套用到全样本——系数只用早期数据,不泄漏 OOS
    fit = (yrs <= 2020) & np.isfinite(base)
    for k, v in raw.items():
        m = fit & np.isfinite(v); A = np.c_[np.ones(m.sum()), base[m]]; coef = np.linalg.lstsq(A, v[m], rcond=None)[0]
        ALL[k] = v - (coef[0] + coef[1] * base)
    ALL.to_pickle(DATA / "_B0_features2.pkl")
    cand = DUR + MIC + BRP + VOLO + ["relvol", "bo_strength", "z_open", "netvol15", "vwap_dev"]       # 最后 5 个 = F1 里的对照
    periods = {"dev(2021-22)": (yrs >= 2021) & (yrs <= 2022), "test(2023-26)": yrs >= 2023}
    rows = []
    for f in cand:
        x = ALL[f].to_numpy(float)
        for sd, sm in (("全部", np.ones(len(y), bool)), ("多", side > 0), ("空", side < 0)):
            r = {"feature": f, "side": sd}
            for pn, pm in periods.items():
                ic, lo, hi, n = cluster_boot_ic(x[sm & pm], y[sm & pm], day_id[sm & pm]); r[f"ic_{pn}"] = ic; r[f"lo_{pn}"] = lo; r[f"hi_{pn}"] = hi
            rows.append(r)
    R = pd.DataFrame(rows)
    sgn_ok = lambda a, b_, la, lb, ha, hb: (np.sign(a) == np.sign(b_)) & (a != 0)
    R["同号"] = np.sign(R["ic_dev(2021-22)"]) == np.sign(R["ic_test(2023-26)"])
    R["两段区间都不含0"] = ((R["lo_dev(2021-22)"] > 0) | (R["hi_dev(2021-22)"] < 0)) & ((R["lo_test(2023-26)"] > 0) | (R["hi_test(2023-26)"] < 0))
    R.to_csv(TAB / "b0_univariate_ic.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    show = R.copy()
    for c in ("ic_dev(2021-22)", "ic_test(2023-26)"): show[c] = show[c].round(3)
    print("1) 单变量 IC(OOS;Spearman(特征, y);y=入场→离场方向收益 bp;自助按交易日聚类)\n")
    print(show[["feature", "side", "ic_dev(2021-22)", "ic_test(2023-26)", "同号", "两段区间都不含0"]].to_string(index=False))
    print("\n=> 两段都显著且同号的(feature, side):")
    print(R[R["同号"] & R["两段区间都不含0"]][["feature", "side", "ic_dev(2021-22)", "ic_test(2023-26)"]].round(3).to_string(index=False))
    print(f"\n检验总数 {len(R)}(特征×方向);随机期望'两段都显著且同号'的数量 ≈ {len(R) * 0.05 * 0.05 * 0.5 * 2:.1f}(双侧 5%×5%,同号)")


if __name__ == "__main__":
    main()
