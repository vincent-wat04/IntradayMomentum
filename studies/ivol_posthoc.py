"""事后(看到 ivol_objective / ivol_entry 结果后追加;不改变预注册判定;累计 +1):
  1) 组合日内已实现方差的分解:QQQ 腿 / Sleeve 腿 / 交叉项各占多少 —— 解释 O4 为什么几乎不改变日内波动
  2) 单特征规则:跳过 rv_surp 位于因果分位最高 20% 的笔(rv_surp = 截至决策时点的累计 RV ÷ 过去 20 日同一时点均值,取对数)
     单格置换 p(年内逐笔置换 1000 次)
python3 ivol_posthoc.py
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
import ivol_objective_study as A
import sizing_sigma_feed as SF
from b0_econ import causal_rank, panel
from b0_validate import Ctx
from etft.intraday.bars import load_bars


def main():
    pd.set_option("display.width", 240, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; I = SF.build_inputs(b); Q, _ = A.build_qv(b, I)
    m = (days >= "2021-01-01") & (days.dayofweek != 0) & (b.last_idx.to_numpy() >= 389)
    q = Q[m]; tot = (q.cc + q.LL + q.SS + 2 * q.EL + 2 * q.ES + 2 * q.LS).sum()
    print("=== 1. 组合日内已实现方差的分解(k≡1,OOS 2021–26)===")
    for nm, v in (("QQQ 腿(含 CPI overlay)", q.cc), ("Sleeve 多头腿", q.LL), ("Sleeve 空头腿", q.SS), ("QQQ×多头 交叉", 2 * q.EL), ("QQQ×空头 交叉", 2 * q.ES), ("多×空 交叉", 2 * q.LS)):
        print(f"  {nm:22s} {v.sum() / tot:7.1%}")
    # ---- 2
    F = pd.read_pickle(SF.OUT.parent.parent.parent / "data" / "_IVT_features.pkl") if False else pd.read_pickle(Path(os.environ.get("ETFT_DATA", "../data")) / "_IVT_features.pkl")
    ctx = Ctx(); x = -F.rv_surp.to_numpy()                                # 预测值越低越该跳过 → 取负号,跳过 rv_surp 最高的 20%
    rk = causal_rank(x, ctx.day_arr); c = ctx.rule_curve(x, rk, "skip", 0.20)
    obs = {k: ctx.dsr(c, k) for k in ("dev", "test", "full")}; obs["组合 全"] = ctx.dsr(c, "full", port=True)
    rg = np.random.default_rng(7); nul = []
    for _ in range(1000):
        perm = np.arange(len(x))
        for yv in np.unique(ctx.year):
            ix = np.where(ctx.year == yv)[0]; perm[ix] = rg.permutation(ix)
        xp = x[perm]; nul.append(ctx.dsr(ctx.rule_curve(xp, causal_rank(xp, ctx.day_arr), "skip", 0.20), "full"))
    p = (np.sum(np.array(nul) >= obs["full"]) + 1) / 1001
    print("\n=== 2. 跳过 rv_surp 最高 20% 的笔(事后单格)===")
    print(f"  Sleeve ΔSR dev(21–22) {obs['dev']:+.3f} / test(23–26) {obs['test']:+.3f} / 全OOS {obs['full']:+.3f};组合 ΔSR {obs['组合 全']:+.3f};单格置换 p = {p:.3f}")
    for sd_, nm in ((1, "只跳多头"), (-1, "只跳空头")):
        keep = ~(np.isfinite(x) & (rk <= 0.20) & (ctx.side == sd_)); cc = ctx.curve(keep)
        print(f"  {nm}: dev {ctx.dsr(cc, 'dev'):+.3f} / test {ctx.dsr(cc, 'test'):+.3f} / 全 {ctx.dsr(cc, 'full'):+.3f}")
    sk = np.isfinite(x) & (rk <= 0.20)
    print(f"  被跳过的笔:均值 {ctx.pnl[sk & (ctx.year >= 2021)].mean() * 1e4:+.2f}bp(保留 {ctx.pnl[~sk & (ctx.year >= 2021)].mean() * 1e4:+.2f}bp);占 OOS 笔数 {sk[ctx.year >= 2021].mean():.1%}")
    mm = ctx.per_mask["full"]
    print(pd.DataFrame({"不过滤": panel(pd.Series(ctx.base[mm])), "跳过 rv_surp 高 20%": panel(pd.Series(c[mm]))}).T.round(3).to_string())


if __name__ == "__main__":
    main()
