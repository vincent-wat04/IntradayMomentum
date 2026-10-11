"""因子/横截面层审计：截断不变性 + 对齐的预言机对照。

两个核心手段：
  A. 截断不变性：用截断到 T 的数据算因子，和用全历史算的因子在 T 这一天比。
     任何前视（居中窗口、全样本标准化、bfill、未来收益泄漏）都会破坏相等。
  B. 预言机对照：往回测引擎里灌入两种人造信号 ——
       oracle    score_t = r_{t+1}（真的能预测明天）-> 必须赚大钱（证明测试有鉴别力）
       contemp   score_t = r_t    （只有今天收盘才知道，对明天没信息）-> 必须 ≈ 0
     如果 contemp 赚钱，说明引擎把 score_t 用在了 r_t 上，即泄漏。
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
import report_protocol as RP
import xs_factor_library as XL

OUT = Path("outputs/full/audit"); OUT.mkdir(parents=True, exist_ok=True)
RESULTS = []


def check(name, ok, detail, catches):
    RESULTS.append(dict(check=name, pass_=bool(ok), detail=str(detail)[:400], catches=catches))
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}")
    print(f"         {detail}")
    return ok


def load_small():
    d = Path("data/xs")
    px, op, hi, lo, vo = [pd.read_csv(d / f"{k}.csv", index_col=0, parse_dates=True)
                          for k in ("close", "open", "high", "low", "volume")]
    return px, op, hi, lo, vo


def main():
    px, op, hi, lo, vo = load_small()
    print(f"横截面面板：{px.shape[1]} 只，{px.shape[0]} 天，{px.index[0].date()} .. {px.index[-1].date()}")

    # ================================================== A. 截断不变性
    print("\n=== A. 因子库截断不变性 ===")
    F_full, r_full = XL.build(px, op, hi, lo, vo)
    names = sorted(F_full)
    print(f"  因子库产出 {len(names)} 个因子")
    for frac in (0.55, 0.75, 0.92):
        T = px.index[int(len(px) * frac)]
        sl = slice(None, T)
        F_tr, _ = XL.build(px.loc[sl], op.loc[sl], hi.loc[sl], lo.loc[sl], vo.loc[sl])
        bad, checked = {}, 0
        for k in names:
            if k not in F_tr or T not in F_tr[k].index or T not in F_full[k].index:
                continue
            a, b = F_tr[k].loc[T], F_full[k].loc[T]
            both = a.notna() & b.notna()
            if both.sum() < 5:
                continue
            checked += 1
            rel = ((a[both] - b[both]).abs() / (1e-12 + b[both].abs())).max()
            nanmis = int((a.isna() != b.isna()).sum())
            if rel > 1e-8 or nanmis > 0:
                bad[k] = dict(max_rel=float(rel), nan_mismatch=nanmis)
        check(f"A.1 截断在 {T.date()}（{frac:.0%} 处）：{checked} 个因子的当日值不变",
              not bad, (f"{len(bad)} 个因子被未来数据影响：{list(bad)[:8]}" if bad
                        else f"{checked} 个因子全部逐值一致（容差 1e-8）"),
              "居中窗口/全样本标准化/bfill/用未来收益构造因子")

    # 前向收益本身必须**依赖**未来（反向对照，证明截断测试有鉴别力）
    T = px.index[int(len(px) * 0.75)]
    f_full = RP.fwd_ret(px, 20).loc[T]
    f_tr = RP.fwd_ret(px.loc[:T], 20).loc[T]
    check("A.2 反向对照：前向收益必须在截断后变成 NaN（否则截断没生效）",
          bool(f_tr.isna().all() and f_full.notna().any()),
          f"截断后 {int(f_tr.notna().sum())} 个非空（应为 0），全历史 {int(f_full.notna().sum())} 个非空",
          "截断测试本身失效")

    # 预处理步骤的截断不变性
    f0 = F_full[names[0]]
    size = np.log1p(px * vo.rolling(252, min_periods=60).mean())
    sec = pd.Series({c: ["A", "B", "C", "D"][i % 4] for i, c in enumerate(px.columns)})
    for fn, lab in ((lambda d: RP.winsor_std(d), "winsor_std 去极值标准化"),
                    (lambda d: RP.neutral_ind_size(d, sec, size.reindex(d.index)),
                     "neutral_ind_size 行业市值中性化")):
        a = fn(f0.loc[:T]).loc[T]
        b = fn(f0).loc[T]
        both = a.notna() & b.notna()
        rel = ((a[both] - b[both]).abs() / (1e-12 + b[both].abs())).max() if both.any() else 0
        check(f"A.3 {lab} 是逐横截面的（不含时序信息）", rel < 1e-8,
              f"最大相对差 {rel:.2e}，比较 {int(both.sum())} 只",
              "用全样本分位数去极值、用全样本均值标准化")

    # ================================================== B. 对齐的预言机对照
    print("\n=== B. 回测对齐（预言机对照）===")
    r = px.pct_change()
    sig_oracle = r.shift(-1)          # score_t = r_{t+1}，真预言机
    sig_contemp = r.copy()            # score_t = r_t，今天收盘才知道，对明天无信息
    sig_lagged = r.shift(1)           # score_t = r_{t-1}，更旧的信息
    for nm, s, expect in (("oracle score_t=r_{t+1}", sig_oracle, "必须 >> 0"),
                          ("contemp score_t=r_t", sig_contemp, "必须 ≈ 0"),
                          ("lagged score_t=r_{t-1}", sig_lagged, "必须 ≈ 0")):
        for P in (1, 5):
            _, ls, _ = RP.group_books(s, r, P=P, fee_bp=0.0)
            a = RP.ann(ls)
            sr = float(ls.dropna().mean() / ls.dropna().std() * np.sqrt(252))
            if "oracle" in nm:
                ok = sr > 3
            else:
                ok = abs(sr) < 1.5
            check(f"B.1 {nm}, P={P}：{expect}", ok,
                  f"多空 Sharpe {sr:+.3f}，年化 {a:+.2%}", "score 与 r 的对齐错位（泄漏或错位一天）")

    # fwd_ret 的精确定位：fwd_ret(h=1) 在 t 这一天应当等于 r_{t+2}
    f1 = RP.fwd_ret(px, 1)
    ics = {k: float(RP.weekly_ic(r.shift(-k), f1, P=5).mean()) for k in range(4)}
    best = max(ics, key=lambda k: ics[k])
    check("B.2 fwd_ret(h=1) 精确等于 r_{t+2}（即信号 t -> 次日收盘建仓 -> 再持有一天）",
          best == 2 and ics[2] > 0.999,
          f"用 r_{{t+k}} 做预言机的 IC：" + ", ".join(f"k={k}:{v:+.4f}" for k, v in ics.items())
          + f" -> 最高在 k={best}",
          "fwd_ret 的 shift 方向错、差一天")

    # ---- group_books 与 fwd_ret 的滞后口径**不一致**：必须量化而不是忽略
    gb_ics = {}
    for k in range(3):
        _, ls, _ = RP.group_books(r.shift(-k), r, P=1, fee_bp=0.0)
        s = ls.dropna()
        gb_ics[k] = float(s.mean() / s.std() * np.sqrt(252))
    gb_best = max(gb_ics, key=lambda k: gb_ics[k])
    check("B.3 已知口径不一致：group_books 配 r_{t+1}（收盘建仓），fwd_ret 配 r_{t+2}（次日建仓）",
          True,
          f"group_books 的预言机 Sharpe：" + ", ".join(f"r_{{t+{k}}}:{v:+.1f}" for k, v in gb_ics.items())
          + f" -> 峰值在 r_{{t+{gb_best}}}。即 IC 用保守口径、组合回测用乐观口径，差一天。"
          f"两者都**没有泄漏**（信号不含未来），只是执行假设不同。",
          "文档化一个真实存在的口径不一致（本次审计发现）")

    # 量化这个不一致对结论的影响
    sec5 = pd.Series({c: ["A", "B", "C", "D", "E"][i % 5] for i, c in enumerate(px.columns)})

    def gb_lag(score, rr, P, extra):
        rk = score.rank(axis=1, pct=True)
        bnd = pd.Series(np.arange(len(score)) % P == 0, index=score.index)
        o = {}
        for g in (1, 5):
            w = ((rk > (g - 1) / 5) & (rk <= g / 5)).astype(float)
            w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
            held = (w.where(bnd, np.nan).ffill(limit=P - 1) if P > 1 else w)\
                .shift(1 + extra).fillna(0.0)
            o[g] = (held * rr).sum(axis=1) - 12e-4 * held.diff().abs().sum(axis=1)
        return o[5] - o[1]

    rows = []
    for k in names:
        s = RP.winsor_std(RP.neutral_ind_size(
            F_full[k].rolling(20, min_periods=10).mean(), sec5, size))
        for P in (5, 20):
            a = gb_lag(s, r, P, 0).dropna(); bb = gb_lag(s, r, P, 1).dropna()
            if len(a) < 200:
                continue
            rows.append(dict(factor=k, P=P,
                             sr_opt=float(a.mean() / a.std() * np.sqrt(252)),
                             sr_cons=float(bb.mean() / bb.std() * np.sqrt(252))))
    LAG = pd.DataFrame(rows); LAG["delta"] = LAG.sr_cons - LAG.sr_opt
    LAG.to_csv(OUT / "audit_lag_sensitivity.csv", index=False)
    worst = LAG.groupby("P").delta.mean().abs().max()
    top5 = {P: (LAG[LAG.P == P].nlargest(5, "sr_opt").sr_opt.mean(),
                LAG[LAG.P == P].nlargest(5, "sr_opt").sr_cons.mean()) for P in (5, 20)}
    check("B.4 口径不一致对结论的影响可忽略（|平均 ΔSharpe| < 0.05）", worst < 0.05,
          f"{len(names)} 个因子：P=5 平均 Δ {LAG[LAG.P==5].delta.mean():+.3f}（std "
          f"{LAG[LAG.P==5].delta.std():.3f}），P=20 平均 Δ {LAG[LAG.P==20].delta.mean():+.3f}。"
          f"最强 5 个因子：P=5 {top5[5][0]:+.3f}->{top5[5][1]:+.3f}，"
          f"P=20 {top5[20][0]:+.3f}->{top5[20][1]:+.3f}",
          "量化 B.3 的口径不一致是否会推翻 PART6/PART7 的横截面结论")

    # ================================================== C. 中性化是否真的生效
    print("\n=== C. 中性化的有效性 ===")
    sz = RP.winsor_std(size).reindex(f0.index)
    raw = RP.winsor_std(f0)
    neu = RP.winsor_std(RP.neutral_ind_size(f0, sec, size.reindex(f0.index)))
    c_raw = raw.corrwith(sz, axis=1).mean()
    c_neu = neu.corrwith(sz, axis=1).mean()
    check("C.1 市值中性化后与 log 市值的横截面相关降到 ~0", abs(c_neu) < 0.05,
          f"中性化前 corr(因子, 市值) = {c_raw:+.4f}，中性化后 = {c_neu:+.4f}",
          "PART4 的'中性化控制变量又被当成特征喂回去'那一类：中性化被静默撤销")

    R = pd.DataFrame(RESULTS)
    R.to_csv(OUT / "audit_factors.csv", index=False)
    npass, ntot = int(R.pass_.sum()), len(R)
    print(f"\n{'='*70}\n因子/横截面层审计：{npass}/{ntot} 通过")
    if npass < ntot:
        print(R[~R.pass_][["check", "detail"]].to_string(index=False))
    return R


if __name__ == "__main__":
    main()
