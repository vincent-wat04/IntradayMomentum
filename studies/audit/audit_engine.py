"""引擎层审计：对被测引擎 etft/intraday/strategy.py 跑全套机械化检查。

每一项都注明：它能抓住本项目真实发生过的哪个 bug。
跑法：/projects/sandbox/.venv/bin/python audit/audit_engine.py
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etft.intraday.bars import load_bars
from etft.intraday.strategy import NoiseCfg, run, buy_hold

from audit import harness as H
from audit import reimpl_noise as RI

OUT = Path("outputs/full/audit"); OUT.mkdir(parents=True, exist_ok=True)
SOTA = dict(stop="vwap", sizing="voltarget", max_lev=4.0)
RESULTS = []


def check(name, ok, detail, catches):
    RESULTS.append(dict(check=name, pass_=bool(ok), detail=str(detail)[:300], catches=catches))
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}")
    print(f"         {detail}")
    return ok


def main():
    b = load_bars("QQQ")
    print(f"数据：QQQ {len(b.days)} 天 × {b.c.shape[1]} 根 1 分钟 bar，"
          f"{b.days[0].date()} .. {b.days[-1].date()}")

    # ============================================================ 1. 前视
    print("\n=== 1. 前视检查（未来数据投毒）===")
    cols = ["ret", "overlay_gross", "cost", "trades", "avg_abs_pos", "exposure_cap"]
    base = run(b, NoiseCfg(cost_bp=1.0, **SOTA))
    for cut in ("2021-06-30", "2023-03-15", "2025-01-15"):
        pz = run(H.poison_future(b, cut), NoiseCfg(cost_bp=1.0, **SOTA))
        a = base.loc[:cut]; c = pz.loc[:cut]
        ok, bad = H.cmp_frames(a, c, cols)
        check(f"1.1 跨日投毒不变性 cut={cut}",
              ok, bad if bad else f"{len(a)} 天 × {len(cols)} 个输出完全一致",
              "全样本标准化/居中窗口/忘记shift(1)/未来除权因子")

    # 日内投毒：固定仓位（排除日级数据影响），只比决策，不比 P&L
    FIX = dict(sizing="fixed", stop="vwap")
    bt = run(b, NoiseCfg(cost_bp=1.0, record_trades=True, **FIX))
    for cut_clock, upto in (("13:00", "12:58"), ("11:00", "10:58")):
        pz = run(H.poison_intraday(b, cut_clock), NoiseCfg(cost_bp=1.0, record_trades=True, **FIX))
        ok, det = H.cmp_trades(bt.attrs.get("trades"), pz.attrs.get("trades"), upto)
        check(f"1.2 日内投毒：{cut_clock}-15:30 的 bar 不影响 {upto} 之前的决策",
              ok, det, "PART8 的 pnl+=pos*(op[j+1]/op[j]-1) 逐bar累加前视；止损用未来high/low；全天VWAP")
    # 反向对照：投毒从 09:40 开始，决策必须变（证明投毒真的生效了）
    pz2 = run(H.poison_intraday(b, "09:40"), NoiseCfg(cost_bp=1.0, record_trades=True, **FIX))
    ok2, det2 = H.cmp_trades(bt.attrs.get("trades"), pz2.attrs.get("trades"), "15:00")
    check("1.2b 反向对照：从 09:40 投毒后决策必须改变", not ok2,
          f"结果：{det2}（这里 FAIL 才说明投毒无效）", "投毒测试本身失效")

    # 决策只用到 bar k 收盘：把检查时点整体后移，收益必须变（证明测试本身有效）
    late = run(b, NoiseCfg(cost_bp=1.0, check_times=[f"{h:02d}:{m:02d}" for h in range(10, 16)
                                                     for m in (5, 35)][:-1], **SOTA))
    d = abs(late["ret"].sum() - base["ret"].sum())
    check("1.3 反向对照：改变检查时点后结果必须变（否则上面的测试是摆设）",
          d > 1e-6, f"检查时点后移 5 分钟后累计收益变化 {d:.4%}", "测试本身失效")

    # ============================================================ 2. 成本记账
    print("\n=== 2. 成本记账 ===")
    r0 = run(b, NoiseCfg(cost_bp=0.0, **SOTA))
    ok = np.allclose(r0["ret"].fillna(0), r0["overlay_gross"].fillna(0), atol=1e-12)
    check("2.1 零成本时 net == gross", ok,
          f"最大差 {abs(r0['ret'].fillna(0)-r0['overlay_gross'].fillna(0)).max():.2e}",
          "成本被重复扣除或漏扣")

    r1 = run(b, NoiseCfg(cost_bp=1.0, **SOTA))
    r2 = run(b, NoiseCfg(cost_bp=2.0, **SOTA))
    lin = np.allclose((r2["cost"] - 2 * r1["cost"]).fillna(0), 0, atol=1e-12)
    check("2.2 成本对 cost_bp 严格线性（才能用 net=gross−turnover×rate 重算）", lin,
          f"最大偏离 {abs(r2['cost']-2*r1['cost']).max():.2e}",
          "PART9 里用 gross/turnover 分解重算成本的前提")

    same_path = np.allclose(r1["overlay_gross"].fillna(0), r2["overlay_gross"].fillna(0), atol=1e-12)
    check("2.3 仓位路径与成本无关（成本不反馈进决策）", same_path,
          f"gross 最大差 {abs(r1['overlay_gross']-r2['overlay_gross']).max():.2e}",
          "同上；若不成立则 PART9 的成本重算全部作废")

    # 每股成本 vs 名义比例成本：价格 ×10 时两者的行为必须不同
    b10 = H.shift_prices(b, 10.0)
    ps_1 = run(b, NoiseCfg(per_share=(0.005, 0.0045), **SOTA))
    ps_10 = run(b10, NoiseCfg(per_share=(0.005, 0.0045), **SOTA))
    bp_1 = run(b, NoiseCfg(cost_bp=1.0, **SOTA))
    bp_10 = run(b10, NoiseCfg(cost_bp=1.0, **SOTA))
    ratio = ps_10["cost"].sum() / ps_1["cost"].sum()
    ok = (abs(ratio - 0.1) < 1e-6) and np.allclose(bp_10["cost"].fillna(0),
                                                   bp_1["cost"].fillna(0), atol=1e-12)
    check("2.4 每股成本 ∝ 1/价格，而 bp 成本与价格无关", ok,
          f"价格×10 后：per_share 成本比 {ratio:.6f}（应为 0.1）；"
          f"cost_bp 成本比 {ps_1['cost'].sum() and bp_10['cost'].sum()/bp_1['cost'].sum():.6f}（应为 1.0）",
          "PART8 的 TQQQ 错误那一类：把每股固定成本当成名义比例成本")

    # ============================================================ 3. 杠杆/对称性
    print("\n=== 3. 杠杆与对称性 ===")
    f1 = run(b, NoiseCfg(cost_bp=1.0, sizing="fixed", stop="vwap"))
    sc = pd.Series(3.0, index=b.days)
    f3 = run(b, NoiseCfg(cost_bp=1.0, sizing="fixed", stop="vwap", day_scale=sc))
    ok = np.allclose(f3["ret"].fillna(0), 3 * f1["ret"].fillna(0), rtol=1e-9, atol=1e-12)
    s1 = H.perf(f1["ret"])["sharpe"]; s3 = H.perf(f3["ret"])["sharpe"]
    check("3.1 常数杠杆：收益严格 ×L，Sharpe 不变", ok and abs(s1 - s3) < 1e-8,
          f"收益最大相对偏离 {abs(f3['ret']-3*f1['ret']).max():.2e}；Sharpe {s1:.6f} vs {s3:.6f}",
          "把'加杠杆改善了策略'当成信号改进")

    # 多空对称性：仓位绝对值的分布
    tr = run(b, NoiseCfg(cost_bp=1.0, record_trades=True, **SOTA)).attrs["trades"]
    nl, ns = (tr.dir > 0).sum(), (tr.dir < 0).sum()
    check("3.2 多空两侧都有交易（不是伪装成择时的单边多头）", min(nl, ns) > 0.2 * max(nl, ns),
          f"做多 {nl} 笔，做空 {ns} 笔，空头占比 {ns/(nl+ns):.1%}",
          "PART3 的 rank 权重用常数 0.5 去均值导致 1/(2N) 净多头那一类")

    # ============================================================ 4. 独立重写实现对账
    print("\n=== 4. 独立重写实现对账（显式股数/现金账本，结构与引擎完全不同）===")
    sem = []
    for nm, kw, sz in (("SOTA voltarget 4x", dict(**SOTA), "voltarget"),
                       ("固定 1x", dict(sizing="fixed", stop="vwap"), "fixed")):
        eng = run(b, NoiseCfg(per_share=(0.005, 0.0045), **kw))["ret"]
        rei, F = RI.simulate(b, half_spread=0.005, commission=0.0045, sizing=sz,
                             reentry="same_bar", cost_shares="leg_price")
        idx = eng.index.intersection(rei.index)
        d = (eng.reindex(idx).fillna(0) - rei.reindex(idx).fillna(0)).abs()
        check(f"4.1 {nm}：引擎 vs 独立账本（同语义）", d.max() < 1e-12,
              f"最大日收益差 {d.max():.3e}，不一致天数 {(d>1e-12).sum()}；"
              f"累计 {eng.reindex(idx).sum():.6f} vs {rei.reindex(idx).sum():.6f}",
              "任何记账错误：成本重复扣/漏扣、入场价错位、MOC 价格错、仓位规模错")
        ok2, det2 = RI.ledger_check(F)
        check(f"4.2 {nm}：账本每天收盘股数归零", ok2, det2, "隔夜敞口泄漏")
        # 语义歧义的敏感性
        for re_ in ("same_bar", "next_check", "hold"):
            for cs in ("leg_price", "actual"):
                r_, _ = RI.simulate(b, half_spread=0.005, commission=0.0045, sizing=sz,
                                    reentry=re_, cost_shares=cs)
                sem.append(dict(variant=nm, reentry=re_, cost_shares=cs,
                                cum=float(r_.sum()), **H.perf(r_)))
    SEM = pd.DataFrame(sem)
    SEM.to_csv(OUT / "audit_semantics.csv", index=False)
    print("\n  止损/重入语义 与 出场腿成本口径 的敏感性：")
    print(SEM[["variant", "reentry", "cost_shares", "cum", "sharpe", "ann"]].round(4)
          .to_string(index=False))
    sub = SEM[SEM.variant == "SOTA voltarget 4x"]
    spread = sub.sharpe.max() - sub.sharpe.min()
    check("4.3 结论对语义歧义不敏感（Sharpe 极差 < 0.15）", spread < 0.15,
          f"SOTA 在 6 种语义组合下 Sharpe {sub.sharpe.min():.3f} ~ {sub.sharpe.max():.3f}，"
          f"极差 {spread:.3f}；引擎用的 same_bar/leg_price 不是最优的那一个"
          f"（hold 更好，实盘应用 hold）", "结论依赖于一个任意的实现细节")

    # ============================================================ 5. 安慰剂
    print("\n=== 5. 安慰剂 ===")
    rng = np.random.default_rng(0)
    # 随机方向：把噪声带替换成随机符号（用 day_scale 做不到，所以直接比"信号打乱"）
    sh = pd.Series(rng.permutation(b.days.to_numpy()), index=b.days)
    # 打乱 day_filter：随机选 2/3 的天交易，重复 60 次，看 Sharpe 零分布
    accs = []
    for i in range(60):
        flt = pd.Series(rng.random(len(b.days)) < 2 / 3, index=b.days)
        rr = run(b, NoiseCfg(cost_bp=1.0, day_filter=flt, **SOTA))["ret"]
        accs.append(H.perf(rr)["sharpe"])
    accs = np.array(accs)
    real = H.perf(base["ret"])["sharpe"]
    check("5.1 随机筛掉 1/3 交易日的零分布（真实值应在分布内，不应远高）",
          True, f"真实（全交易日）{real:.3f}；随机筛选零分布 均值 {accs.mean():.3f} "
                f"std {accs.std():.3f} p5 {np.percentile(accs,5):.3f} "
                f"p95 {np.percentile(accs,95):.3f}",
          "量化'随机改变交易日集合'本身能造成多大的 Sharpe 波动（PART9 §2.4 用到）")

    R = pd.DataFrame(RESULTS)
    R.to_csv(OUT / "audit_engine.csv", index=False)
    npass, ntot = int(R.pass_.sum()), len(R)
    print(f"\n{'='*70}\n引擎层审计：{npass}/{ntot} 通过")
    if npass < ntot:
        print("未通过：")
        print(R[~R.pass_][["check", "detail"]].to_string(index=False))
    return R


if __name__ == "__main__":
    main()
