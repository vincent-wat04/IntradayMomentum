"""v3 ① 组合(QQQ 腿 RV21×SMA200 + v3 Sleeve)的两个仓位模型变体回测(描述性,事后配置,不改变 v3 判定):
  V3      现行:sigmoid PPP,CLEAN 7 特征(tearsheet_v3 ①)
  V3-L    F1 线性链接 k = clip(1+θᵀz, 0, 2),CLEAN 7 特征
  V3-LD   线性链接 + CLEAN + dur_rate(10:00 前价格持续期事件率)
其余完全同 v3(FOMC≥14:30、CPI 回补 1×、FFA-P 跳过 20%、排除周一、15:40 平仓、走步重训 2021 起生效)。
与 tearsheet_v3.py 同口径:全部交易日、tearsheet.metrics、dev 2019–22 / test 2023–26 / 全 2019–26。
python3 tearsheet_v3_variants.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_pred_study as DP
import sizing_models_study as M
import sizing_sigma_feed as SF
import tearsheet as TS
import unified_sizing_posthoc as P
import unified_sizing_study as U
from cjkfont import use_cjk
from etft.intraday.bars import load_bars

KEYS = ["年化收益 CAGR", "年化波动", "Sharpe（绝对）", "Sortino", "最大回撤 MDD", "Calmar", "最长水下（交易日）", "日胜率", "盈亏比（日）",
        "最好单日", "最差单日", "偏度", "超额峰度", "CVaR95（日）", "β（对 QQQ）", "α（年化）", "α 的 t 值 (HAC)"]
PER = (("dev 2019–22", "2019", "2022"), ("test 2023–26", "2023", "2026"), ("OOS 2021–26", "2021", "2026"), ("全 2019–26", "2019", "2026"))


def main():
    pd.set_option("display.width", 260, "display.max_rows", 200, "display.float_format", lambda v: f"{v:9.4f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); L2, S2, GL, GS, keep, ov, X, leg1, bh = (I[k] for k in ("L2", "S2", "GL", "GS", "keep", "ov", "X", "leg1", "bh"))
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X = X.join(M.family_features(b, X))
    Lb, Sb, Eb = (L2 * 1e4).to_numpy(), (S2 * 1e4).to_numpy(), ((leg1 + ov) * 1e4).to_numpy()
    fit = lambda kd: (lambda Z, l_, s_, e_, lam: M.ppp_fit(Z, l_, s_, e_, lam, kd))
    K = {"V3 现行(sigmoid)": M.wf(X, P.CLEAN, Lb, Sb, Eb, sel, days, fit("sig"), U.LAMS)[0],
         "V3-L 线性": M.wf(X, P.CLEAN, Lb, Sb, Eb, sel, days, fit("lin"), U.LAMS)[0],
         "V3-LD 线性+dur_rate": M.wf(X, P.CLEAN + ["dur_rate"], Lb, Sb, Eb, sel, days, fit("lin"), U.LAMS)[0]}
    def sleeve(k):
        kl = k.kl.where(om, 1.0).fillna(1.0); ks = k.ks.where(om, 1.0).fillna(1.0)
        return kl * L2 + ks * S2 + ov - (kl * GL + ks * GS) * keep
    series = {"QQQ 买入持有": bh}
    for nm, k in K.items(): series[f"组合 | {nm}"] = leg1 + sleeve(k)
    for nm, k in K.items(): series[f"Sleeve | {nm}"] = sleeve(k)
    assert all(int((s.loc[s.index.dayofweek == 0] != 0).sum()) == 0 for nm, s in series.items() if nm.startswith("Sleeve")), "Sleeve 必须排除周一"
    rows = []
    for pn, a, z in PER:
        for nm, x in series.items():
            m = TS.metrics(x.loc[a:z], bench=None if nm == "QQQ 买入持有" else bh.loc[a:z]); rows.append(dict(期间=pn, 序列=nm, **{k_: m.get(k_, np.nan) for k_ in KEYS}))
    T = pd.DataFrame(rows)
    for pn, _, _ in PER:
        print(f"\n===== {pn} =====")
        print(T[T.期间 == pn].set_index("序列")[KEYS].T.to_string())
    yr = pd.DataFrame({nm: {y: TS.metrics(x.loc[str(y)])["Sharpe（绝对）"] for y in range(2019, 2027)} for nm, x in series.items() if not nm.startswith("Sleeve")})
    print("\n===== 分年 Sharpe(组合)====="); print(yr.round(3).to_string())
    T.to_csv(M.TAB / "tearsheet_v3_variants.csv", index=False)
    # ---- 图
    use_cjk(plt)
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    col = {"QQQ 买入持有": "black", "组合 | V3 现行(sigmoid)": "tab:green", "组合 | V3-L 线性": "tab:blue", "组合 | V3-LD 线性+dur_rate": "tab:red"}
    for nm, cc in col.items():
        x = series[nm].loc["2019":]; eq = (1 + x).cumprod(); te = TS.metrics(x.loc["2023":])
        ax[0].plot(eq, color=cc, lw=2 if nm != "QQQ 买入持有" else 1.4, ls=":" if nm == "QQQ 买入持有" else "-",
                   label=f"{nm}  (test Sharpe {te['Sharpe（绝对）']:.2f}, 全期 MDD {TS.metrics(x)['最大回撤 MDD']:.1%})")
        dd = eq / eq.cummax() - 1; ax[1].plot(dd, color=cc, lw=1.2, ls=":" if nm == "QQQ 买入持有" else "-")
    for a_ in ax: a_.axvline(pd.Timestamp("2023-01-01"), color="gray", ls="--", lw=1); a_.grid(alpha=.3)
    ax[0].axvline(pd.Timestamp("2021-01-01"), color="gray", ls=":", lw=1)
    ax[0].set_yscale("log"); ax[0].set_title("v3 ① 组合:仓位模型变体(2021 起走步生效;2023 起 test)", loc="left"); ax[0].legend(fontsize=9, loc="upper left")
    ax[1].set_title("回撤", loc="left")
    out = Path("../outputs/figs"); out.mkdir(parents=True, exist_ok=True); fig.savefig(out / "tearsheet_v3_variants.png", dpi=120, bbox_inches="tight")
    print("图:outputs/figs/tearsheet_v3_variants.png")


if __name__ == "__main__":
    main()
