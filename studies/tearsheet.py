"""策略 tearsheet：净值 / 回撤 / 月度 / 滚动 Sharpe / 完整指标表。

对象是当前实盘配置 ⑦ = 多空 + band止损 + 硬尾k=0.5 + 排除周一，
30 相位均值、TQQQ 单股成本、不做任何事后挑选。

三条线的含义（给客户看必须讲清楚）：
  * **Sleeve**      策略本身（日内多空，隔夜不留仓，β≈0）
  * **QQQ B&H**     基准（全天持有，含股息）
  * **组合 1:1**    QQQ B&H + 1.0x Sleeve —— 这才是实际产品形态
Sharpe 对规模齐次，所以无法靠加杠杆改善 QQQ 的 Sharpe；
sleeve 的价值在于 β≈0、与基准相关性≈0，是分散化而不是替代。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjkfont import use_cjk
from combo_lib import DEFAULT_HI, DEFAULT_LO, qqq_leg

use_cjk(plt)
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("outputs/full/intraday")
FIG = Path("outputs/full/figs")
VER = "多空 band+尾0.5 无周一"
LBL = "Sleeve（实盘配置 band+尾0.5・排除周一）"
CMB = "组合 = QQQ + 1.0×Sleeve"
RVC = f"组合+RV降风险 (QQQ腿 lo={DEFAULT_LO:g}, hi={DEFAULT_HI:g})"


def metrics(x, bench=None, rf=0.0):
    x = pd.Series(x).dropna()
    n = len(x)
    eq = (1 + x).cumprod()
    dd = eq / eq.cummax() - 1
    cagr = float(eq.iloc[-1] ** (252 / n) - 1)
    vol = float(x.std() * np.sqrt(252))
    neg = x[x < 0]
    dsd = float(neg.std() * np.sqrt(252)) if len(neg) > 2 else np.nan
    mdd = float(dd.min())
    # 最长水下天数
    under = (dd < -1e-12).astype(int)
    longest = cur = 0
    for v in under:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    gains, losses = x[x > 0].sum(), -x[x < 0].sum()
    m = dict()
    m["总收益"] = float(eq.iloc[-1] - 1)
    m["年化收益 CAGR"] = cagr
    m["年化波动"] = vol
    m["Sharpe（绝对）"] = float(x.mean() / x.std() * np.sqrt(252))
    m["Sortino"] = cagr / dsd if dsd and dsd == dsd else np.nan
    m["最大回撤 MDD"] = mdd
    m["Calmar"] = cagr / abs(mdd) if mdd < 0 else np.nan
    m["最长水下（交易日）"] = longest
    m["日胜率"] = float((x[x != 0] > 0).mean())
    m["盈亏比（日）"] = float(gains / losses) if losses > 0 else np.nan
    m["最好单日"] = float(x.max())
    m["最差单日"] = float(x.min())
    m["偏度"] = float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    m["超额峰度"] = float(((x - x.mean()) ** 4).mean() / x.std() ** 4 - 3)
    m["VaR95（日）"] = float(x.quantile(0.05))
    m["CVaR95（日）"] = float(x[x <= x.quantile(0.05)].mean())
    if bench is not None:
        bb = pd.Series(bench).reindex(x.index).fillna(0.0)
        X = sm.add_constant(bb.to_numpy())
        r = sm.OLS(x.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        a_d, beta = float(r.params[0]), float(r.params[1])
        resid = x.to_numpy() - X @ r.params
        sd_e = float(np.std(resid, ddof=2))
        m["β（对 QQQ）"] = beta
        m["α（年化）"] = a_d * 252
        m["α 的 t 值 (HAC)"] = float(r.tvalues[0])
        m["alpha Sharpe"] = a_d / sd_e * np.sqrt(252) if sd_e > 0 else np.nan
        m["与 QQQ 相关性"] = float(np.corrcoef(x, bb)[0, 1])
    return m


def fmt(k, v):
    if v != v:
        return "—"
    if k in ("最长水下（交易日）",):
        return f"{int(v)}"
    if k in ("Sharpe（绝对）", "Sortino", "Calmar", "盈亏比（日）", "偏度",
             "超额峰度", "β（对 QQQ）", "α 的 t 值 (HAC)", "alpha Sharpe",
             "与 QQQ 相关性"):
        return f"{v:,.3f}"
    return f"{v:,.2%}"


def main():
    cur = pd.read_pickle(OUT / "strategy_comparison_curves.pkl")
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna()
    s = cur[VER].reindex(bh.index).fillna(0.0)
    combo = bh + 1.0 * s
    leg, wleg = qqq_leg(bh, q=1.0)                 # QQQ 腿按 RV21 百分位调权（已扣 2bp 换手成本）
    combo_rv = leg + 1.0 * s
    series = {LBL: s, "QQQ 买入持有": bh, CMB: combo, RVC: combo_rv}

    # ---------- 指标表（全样本 / dev / test）
    rows = {}
    for nm, x in series.items():
        rows[nm] = metrics(x, bench=bh if nm != "QQQ 买入持有" else None)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "tearsheet_metrics.csv")

    print("=" * 112)
    print("【全样本 2019-01 ~ 2026-10】")
    print("=" * 112)
    hdr = f"{'指标':<20}" + "".join(f"{c:>30}" for c in T.columns)
    print(hdr); print("-" * 112)
    for k in T.index:
        print(f"{k:<20}" + "".join(f"{fmt(k, T.loc[k, c]):>30}" for c in T.columns))

    print("\n" + "=" * 112)
    print("【分期 Sharpe / CAGR / MDD】")
    print("=" * 112)
    print(f"{'期间':<16}{'Sleeve SR':>11}{'QQQ SR':>9}{'组合 SR':>10}"
          f"{'Sleeve CAGR':>13}{'组合 CAGR':>11}{'Sleeve MDD':>12}{'QQQ MDD':>10}{'组合 MDD':>11}")
    for pn, per in (("dev 2019-2022", ("2019", "2022")),
                    ("test 2023-2026", ("2023", "2026")),
                    ("全 2019-2026", None)):
        sl = s if per is None else s.loc[per[0]:per[1]]
        bb = bh if per is None else bh.loc[per[0]:per[1]]
        cb = combo if per is None else combo.loc[per[0]:per[1]]
        a, b, c = metrics(sl), metrics(bb), metrics(cb)
        print(f"{pn:<16}{a['Sharpe（绝对）']:>11.3f}{b['Sharpe（绝对）']:>9.3f}"
              f"{c['Sharpe（绝对）']:>10.3f}{a['年化收益 CAGR']:>13.2%}"
              f"{c['年化收益 CAGR']:>11.2%}{a['最大回撤 MDD']:>12.2%}"
              f"{b['最大回撤 MDD']:>10.2%}{c['最大回撤 MDD']:>11.2%}")

    # ================= 作图 =================
    fig = plt.figure(figsize=(21, 14.5))
    gs = fig.add_gridspec(3, 4, hspace=.40, wspace=.26,
                          width_ratios=[1.0, 1.0, 0.82, 0.95])
    COL = {LBL: "tab:blue", "QQQ 买入持有": "black", CMB: "tab:red", RVC: "tab:cyan"}
    SPLIT = pd.Timestamp("2023-01-01")

    # (1) 净值
    A = fig.add_subplot(gs[0, :2])
    for nm, x in series.items():
        eq = (1 + x).cumprod()
        A.plot(eq, color=COL[nm], lw=2.6 if nm == RVC else (2.2 if "组合" in nm else 1.6),
               ls=":" if nm == "QQQ 买入持有" else "-",
               label=f"{nm}  (Sharpe {metrics(x)['Sharpe（绝对）']:.2f}, "
                     f"CAGR {metrics(x)['年化收益 CAGR']:.1%})")
    A.axvline(SPLIT, color="gray", lw=1.1, ls="--")
    A.set_yscale("log"); A.set_ylabel("累计净值（对数轴，初始=1）")
    A.set_title("累计净值曲线", fontsize=13, loc="left")
    A.legend(fontsize=9.5, loc="upper left"); A.grid(alpha=.3, which="both")
    A.text(SPLIT, A.get_ylim()[0] * 1.04, " 样本外起点 2023", color="gray", fontsize=9)

    # (2) 指标表
    B = fig.add_subplot(gs[0:2, 3]); B.axis("off")
    cells = [[k, fmt(k, T.loc[k, LBL]), fmt(k, T.loc[k, CMB]), fmt(k, T.loc[k, RVC])]
             for k in T.index]
    tb = B.table(cellText=cells, colLabels=["指标", "Sleeve", "组合 1:1", "组合+RV"],
                 cellLoc="right", colLoc="center", loc="upper center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.0); tb.scale(1, 1.42)
    for j in range(4):
        tb[0, j].set_facecolor("#d9d9d9"); tb[0, j].set_text_props(weight="bold")
    for i in range(1, len(cells) + 1):
        tb[i, 3].set_facecolor("#e6f7fa")
    for i in range(1, len(cells) + 1):
        tb[i, 0].set_text_props(ha="left")
    B.set_title("完整指标（全样本 2019-2026）\nβ/α/相关性均为对 QQQ 回归；\n组合内含 QQQ 腿，故 β≈1 不可与 Sleeve 直接比", fontsize=10.5, loc="left")

    # (2b) 分期对照表 —— 固定面板，避免"1.9 还是 1.3"这类期间混淆
    P2 = fig.add_subplot(gs[0, 2]); P2.axis("off")
    prows = []
    for pn, per in (("dev 19-22", ("2019", "2022")),
                    ("test 23-26", ("2023", "2026")),
                    ("全 19-26", None)):
        sel = (lambda x: x) if per is None else (lambda x, per=per: x.loc[per[0]:per[1]])
        prows.append([pn] + [f"{metrics(sel(z))['Sharpe（绝对）']:.3f}" for z in (s, bh, combo, combo_rv)])
    t2 = P2.table(cellText=prows, colLabels=["期间", "Sleeve", "QQQ", "组合1:1", "组合+RV"],
                  cellLoc="center", loc="center")
    t2.auto_set_font_size(False); t2.set_fontsize(9.6); t2.scale(1, 2.0)
    for j in range(5):
        t2[0, j].set_facecolor("#d9d9d9")
    for i in range(1, 4):
        t2[i, 3].set_facecolor("#ffe9e9"); t2[i, 4].set_facecolor("#e6f7fa")
    P2.set_title("Sharpe 分期对照\n（test 期 QQQ 自己 1.50，全样本 0.99）",
                 fontsize=11.5, loc="left", pad=22)

    # (3) 水下回撤
    C = fig.add_subplot(gs[1, :3])
    for nm in (LBL, "QQQ 买入持有", CMB, RVC):
        x = series[nm]; eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
        C.plot(dd * 100, color=COL[nm], lw=1.5,
               ls=":" if nm == "QQQ 买入持有" else "-",
               label=f"{nm}  (MDD {dd.min():.1%})")
        if nm != "QQQ 买入持有":
            C.fill_between(dd.index, dd * 100, 0, color=COL[nm], alpha=.12)
    C.set_ylabel("回撤 (%)"); C.set_title("水下回撤曲线", fontsize=13, loc="left")
    C.legend(fontsize=9.5, loc="lower left"); C.grid(alpha=.3)

    # (4) 日收益分布
    D = fig.add_subplot(gs[2, 2])
    D.hist(s[s != 0] * 1e4, bins=70, color="tab:blue", alpha=.8)
    q5 = s.quantile(.05) * 1e4          # 与指标表同口径：全部交易日（含不开仓的 0）
    D.axvline(q5, color="tab:red", ls="--", lw=1.4,
              label=f"VaR95 {q5:.0f}bp\n（与左表同口径）")
    D.axvline(0, color="k", lw=1)
    D.set_xlabel("日收益 (bp)"); D.set_ylabel("天数")
    D.set_title("Sleeve 日收益分布\n（柱子剔除不开仓的 0，便于看形状）",
                fontsize=11, loc="left")
    D.legend(fontsize=9); D.grid(alpha=.3)

    # (5) 月度热力图
    E = fig.add_subplot(gs[2, :2])
    mo = s.resample("ME").apply(lambda z: (1 + z).prod() - 1)
    M = pd.DataFrame({"y": mo.index.year, "m": mo.index.month, "v": mo.values})
    P = M.pivot(index="y", columns="m", values="v") * 100
    vmax = float(np.nanmax(np.abs(P.to_numpy())))
    im = E.imshow(P.to_numpy(), cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    E.set_xticks(range(12)); E.set_xticklabels([f"{i}月" for i in range(1, 13)], fontsize=9)
    E.set_yticks(range(len(P.index))); E.set_yticklabels(P.index, fontsize=9)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.to_numpy()[i, j]
            if v == v:
                E.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.6)
    E.set_title("Sleeve 月度收益 (%)", fontsize=13, loc="left")
    fig.colorbar(im, ax=E, fraction=.025, pad=.01)

    # (6) 滚动 12 个月 Sharpe
    F = fig.add_subplot(gs[2, 3])
    W = 252
    for nm in (LBL, "QQQ 买入持有"):
        x = series[nm]
        rs = x.rolling(W).mean() / x.rolling(W).std() * np.sqrt(252)
        F.plot(rs, color=COL[nm], lw=1.5,
               ls=":" if nm == "QQQ 买入持有" else "-", label=nm)
    F.axhline(0, color="k", lw=1)
    F.set_ylabel("滚动 12 个月 Sharpe")
    F.set_title("滚动 Sharpe（252 日窗口）", fontsize=12, loc="left")
    F.legend(fontsize=8.5); F.grid(alpha=.3)

    note = ("口径：QQQ 日内噪声区间动量（Zarattini-Aziz-Barbon 2024 框架）；信号用 QQQ、执行用 TQQQ（1/3 名义金额拿同等敞口）。"
            "成本按实测 TQQQ 单腿 0.1323bp 计。\n"
            "所有数字为 30 个检查时点相位的均值（相位整体平移 0~29 分钟），以消除"
            "“相位运气”——论文所用相位是 30 个中的第 90 百分位，单相位结果会系统性偏高。\n"
            "2023 年起为样本外。隔夜不留仓。排除周一：周一期望收益统计上为零"
            "（全样本 −0.81bp，t=−0.23）且不参与最好的 10 个交易日，故剔除近乎无成本。\n"
            "敞口：sleeve 仓位 = min(4, 2%/σ_daily) 倍自有资金，用 TQQQ 以 1/3 名义金额实现；"
            "“组合”线假设 QQQ 满仓之外还有运行 sleeve 的保证金余量，实盘需确认购买力。\n"
            "组合+RV：QQQ 腿每日按 RV21 的 504 日百分位调权 w=0.5+1.0×(1−p)，范围 [0.5,1.5]，年换手约 7 倍，已扣 2bp；"
            "平静期 QQQ 最高 1.5 倍，需要杠杆。参数在看过 2019 年之前的 SPX/NDX 结果后选定，与本窗口有重叠，不是独立样本外；"
            "其价值主要来自 2020 年崩盘（剔除后 ΔSharpe 约缩水 25%~40%）。\n"
            "局限：α 的 HAC t 值在全样本为 2.9，属“可辩护”而非“高置信”；"
            "模拟盘实盘验证自 2026-10-05 起，样本仍短；偏度 +2.7、超额峰度 19.9，"
            "收益高度非正态（右偏厚尾），Sharpe 低估了其尾部特征。")
    fig.text(.012, .004, note, fontsize=8.4, va="bottom", color="#333333")
    fig.suptitle("QQQ 日内噪声区间动量策略 — 业绩总览", fontsize=16, x=0.32, y=.985)
    fig.subplots_adjust(left=.045, right=.985, top=.935, bottom=.105)
    p = FIG / "tearsheet.png"
    fig.savefig(p, dpi=125)
    print(f"\n图 -> {p}\n指标表 -> {OUT}/tearsheet_metrics.csv")


if __name__ == "__main__":
    main()
