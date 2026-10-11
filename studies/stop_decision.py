"""止损最终选型：vwap（当前）vs band+硬尾0.5σ。以及周一那 5 天到底说明了什么。

第一问：长期回测说排除周一更好，但 8 月底以来 5 个周一反而很好 —— 矛盾吗？
第二问：band+硬尾0.5σ 在 dev/test 的每个指标上都不差，为什么不换？

对第二问我之前的理由是"配对检验不显著"。但"不显著"不等于"不该换"：
当两个选项的收益差在噪声内时，就该用**非统计**的标准来选 ——
换手（机械的成本优势）、单笔亏损上限（构造保证）、以及跨 regime 的稳定性。
这个脚本把这些摆开。
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars
from etft.intraday.strategy import buy_hold

OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")


def sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 20 and x.std() > 0 else np.nan


def main():
    b = load_bars("QQQ"); bh = buy_hold(b)
    cur = pd.read_pickle(OUT / "stop_hard_tail_curves.pkl")
    base = cur["vwap（当前基线）"]
    cand = cur["band + 硬尾0.5σ"]
    tailonly = cur["只有硬尾0.5σ"]
    dow = pd.Series(base.index.dayofweek, index=base.index)

    # ---------- 0. 先确认一件事：band+硬尾0.5σ 是不是其实就等于"只有硬尾0.5σ"
    d0 = (cand - tailonly).abs().max()
    print("=" * 100)
    print("0. band+硬尾0.5σ 与 只有硬尾0.5σ 是否同一个东西")
    print("=" * 100)
    print(f"  两条收益曲线最大差 {d0:.2e}  -> "
          f"{'**完全相同**：k=0.5 时硬尾总是先于 band 触发，band 根本没起作用' if d0 < 1e-9 else '不同'}")
    print("  含义：这个配置的正确描述是「**只用逐根 bar 的硬尾止损（0.5σ）**，不用 vwap 也不用 band」")

    # ---------- 1. 周一：5 天能说明什么
    print("\n" + "=" * 100)
    print("1. 周一：长期回测 vs 最近 5 个周一")
    print("=" * 100)
    mon_t = base.loc[TEST[0]:TEST[1]][dow.loc[TEST[0]:TEST[1]] == 0].dropna()
    print(f"  test 期全部周一：{len(mon_t)} 天，日均 {mon_t.mean()*1e4:+.2f}bp，"
          f"标准差 {mon_t.std()*1e4:.1f}bp，t={mon_t.mean()/mon_t.std()*np.sqrt(len(mon_t)):+.2f}")
    se5 = mon_t.std() / np.sqrt(5) * 1e4
    print(f"  -> 任取 5 个周一，其均值的标准误是 {se5:.1f}bp")
    print(f"  -> 离线重放那 5 天的 +54.17bp 距全样本均值 {mon_t.mean()*1e4:+.2f}bp 有 "
          f"{(54.17 - mon_t.mean()*1e4)/se5:+.2f} 个标准误")
    # 这 5 个周一在回测数据里是否也是正的（口径交叉验证）
    recent = base.loc["2026-08-28":].dropna()
    rm = recent[dow.reindex(recent.index) == 0]
    print(f"\n  回测口径下 2026-08-28 起的周一：{len(rm)} 天，"
          f"日均 {rm.mean()*1e4:+.2f}bp，明细 {[f'{d.date()}:{v*1e4:+.0f}bp' for d, v in rm.items()]}")
    print(f"  （离线重放用的是实盘 bar 缓存、回测用的是 min1 面板，两条独立链路）")
    # 那 5 天里有没有一天在主导
    print(f"\n  **分解**：这 4~5 个周一的贡献几乎全来自一天。")
    print(f"  2026-09-21 当天 QQQ 日收益 {float(bh.loc['2026-09-21'])*100:+.2f}%"
          f"（这是个大波动日，策略在大波动日赚钱 —— PART8 §3 的凸性）")
    rm2 = rm.drop(pd.Timestamp("2026-09-21"), errors="ignore")
    print(f"  去掉 2026-09-21 后，其余 {len(rm2)} 个周一日均 {rm2.mean()*1e4:+.2f}bp "
          f"（明细 {[f'{v*1e4:+.0f}' for v in rm2]}）")
    nmon_needed = int(np.ceil((mon_t.std() * 1e4 / (3.57 / 2)) ** 2))
    print(f"\n  要把'周一日均 −3.57bp'这个效应测到 2 个标准误，需要约 {nmon_needed} 个周一"
          f"（≈ {nmon_needed/52:.0f} 年）")

    # ---------- 2. 止损选型：逐年稳定性
    print("\n" + "=" * 100)
    print("2. 止损选型：逐年 Sharpe（跨 regime 稳定性，比全期 Sharpe 更能说明问题）")
    print("=" * 100)
    names = ["vwap（当前基线）", "band + 硬尾0.5σ", "band（对照）",
             "band + 硬尾1.5σ"]
    yr = {}
    for nm in names:
        s = cur[nm]
        yr[nm] = {y: sr(s.loc[f"{y}-01-01":f"{y}-12-31"]) for y in range(2019, 2027)}
    Y = pd.DataFrame(yr).T
    print(Y.round(3).to_string())
    print()
    for nm in names:
        v = np.array([x for x in yr[nm].values() if np.isfinite(x)])
        d_ = sr(cur[nm].loc[DEV[0]:DEV[1]]); t_ = sr(cur[nm].loc[TEST[0]:TEST[1]])
        print(f"  {nm:22s} 逐年均值 {v.mean():+.3f} 标准差 {v.std():.3f}  "
              f"最差年 {v.min():+.3f}  负年数 {(v<0).sum()}/{len(v)}  "
              f"|dev−test| {abs(d_-t_):.3f}")

    # ---------- 3. 非统计的比较口径
    print("\n" + "=" * 100)
    print("3. 收益差在噪声内时，该看的是这些（都不是统计推断）")
    print("=" * 100)
    H = pd.read_csv(OUT / "stop_hard_tail.csv")
    rows = []
    for nm in ("vwap（当前基线）", "band + 硬尾0.5σ"):
        r = H[H.配置 == nm].iloc[0]
        sx = cur[nm].where(dow != 0, 0.0).reindex(bh.index).fillna(0)
        ov = (bh + sx).loc[TEST[0]:TEST[1]].dropna()
        eq = (1 + ov).cumprod(); dd = eq / eq.cummax() - 1
        rows.append(dict(配置=nm, 腿每日=r.腿每日,
                         单笔亏损上限="无（取决于 VWAP 何时触发）" if "vwap" in nm
                         else "≈资金的 1.0%（0.5σ，构造保证）",
                         dev最差日=r.dev_worst, test最差日=r.test_worst,
                         dev年化=r.dev_ann, test年化=r.test_ann,
                         叠加test=sr(ov), 叠加回撤=dd.min()))
    C = pd.DataFrame(rows)
    for c in ("dev最差日", "test最差日", "dev年化", "test年化", "叠加回撤"):
        C[c] = (C[c] * 100).round(2)
    print(C.round(3).to_string(index=False))
    print()
    r0, r1 = H[H.配置 == "vwap（当前基线）"].iloc[0], H[H.配置 == "band + 硬尾0.5σ"].iloc[0]
    print(f"  换手：{r1.腿每日:.2f} vs {r0.腿每日:.2f} 腿/日 -> "
          f"成本低 {(1-r1.腿每日/r0.腿每日):.0%}（**机械优势，不是统计结论**）")
    print(f"  单笔亏损上限：硬尾 0.5σ 给出构造性上限约资金的 1%；"
          f"vwap 只在整半点评估，两点之间无上限")

    # ---------- 4. 多重检验：dev 上的改进会不会是 30 个配置里挑出来的
    print("\n" + "=" * 100)
    print("4. 多重检验：我一共试了约 30 个止损配置，dev 上 +0.096 会不会是挑出来的")
    print("=" * 100)
    devs = []
    for nm, s in cur.items():
        v = sr(s.loc[DEV[0]:DEV[1]])
        if np.isfinite(v):
            devs.append((nm, v))
    devs.sort(key=lambda x: -x[1])
    b0 = sr(base.loc[DEV[0]:DEV[1]])
    print(f"  dev 基线 {b0:+.3f}；{len(devs)} 个配置的 dev Sharpe 分布："
          f"均值 {np.mean([v for _, v in devs]):+.3f}  "
          f"标准差 {np.std([v for _, v in devs]):.3f}  "
          f"最高 {devs[0][1]:+.3f}（{devs[0][0]}）")
    rk = [i for i, (nm, _) in enumerate(devs) if nm == "band + 硬尾0.5σ"][0] + 1
    print(f"  band+硬尾0.5σ 在 dev 上排第 {rk}/{len(devs)}")
    print(f"  -> 它**不是**dev 上的最大值（最大值是 {devs[0][0]}），"
          f"所以不是典型的'挑最大'偏差；但也确实是在同一批里比较出来的")

    print("\n" + "=" * 100)
    print("5. 结论")
    print("=" * 100)
    print("""  收益层面：两者分不出（配对自助 dev p=0.304、test p=0.452）。这一点不变。
  但在"分不出"的前提下，非统计的三条都指向 band+硬尾0.5σ：
    a) 换手低 19%（2.23 vs 2.75 腿/日）—— 机械的成本优势
    b) 单笔亏损有构造性上限（≈资金 1%），vwap 在两个检查点之间完全无上限
    c) 跨 regime 更稳：|dev−test| 更小，逐年负年数更少
  我之前只用"不显著"就否掉它，是把"不显著"误当成"不该换"。
  修正后的建议见对话正文。""")

    Y.to_csv(OUT / "stop_decision_by_year.csv")
    C.to_csv(OUT / "stop_decision_compare.csv", index=False)


if __name__ == "__main__":
    main()
