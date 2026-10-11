"""12 个版本的 alpha / beta 分解：到底是 alpha 还是只是市场敞口？

直接复用 strategy_comparison.py 存下的净值曲线，不重跑模拟。

回归式（日频）：  r_t = α + β·r_bench,t + ε_t

两个基准都报，因为它们回答不同的问题：
  * **全日 B&H**（day_close/prev_close_adj，年化 23.22%、Sharpe 0.994）
    —— 这是叠加决策相关的基准：我们是把 sleeve 叠在一个全天持仓的 QQQ 上。
    策略只做日内、隔夜空仓，所以对这个基准的 β 天然被稀释，这是**对的**。
  * **日内 open→close**（年化 9.30%、Sharpe 0.576，与全日相关 0.78）
    —— 这是策略真正的机会集：用它才能回答"这是不是只是在吃日内市场收益"。

四个指标的定义：
  绝对 Sharpe   mean(r)/std(r)·√252            —— sleeve 自身的全部收益，不做任何对冲
  β             回归斜率                        —— 市场敞口
  α（年化）      α_daily·252                    —— 剔除市场敞口后的超额收益
  **alpha Sharpe**  α_daily/std(ε)·√252        —— 即 appraisal ratio：
                把 β 完全对冲掉之后，剩下那部分的 Sharpe。
                这是"alpha 质量"的正确刻度，和绝对 Sharpe 的差别就是 β 贡献了多少。

α 的 t 值用 Newey-West（HAC, lag=5），避免日频残差自相关把显著性放大。

历史对照（PART2）：当年做过完整 beta 对冲，结论是"**对冲把 sleeve 打死了**"——
本表给出为什么：多空版本的 β 本来就≈0，对冲不掉任何东西，只会白付成本。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars

OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")


def decomp(r, bench, hac=5):
    """返回 (beta, alpha_ann, t_alpha_HAC, alpha_sharpe, abs_sharpe)。"""
    idx = r.index.intersection(bench.index)
    y = r.reindex(idx).fillna(0.0)
    x = bench.reindex(idx).fillna(0.0)
    if len(y) < 60 or y.std() == 0:
        return {}
    X = sm.add_constant(x.to_numpy())
    m = sm.OLS(y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": hac})
    a_d, beta = float(m.params[0]), float(m.params[1])
    resid = y.to_numpy() - X @ m.params
    sd_e = float(np.std(resid, ddof=2))
    return dict(beta=beta, alpha_ann=a_d * 252, t_alpha=float(m.tvalues[0]),
                alpha_sharpe=(a_d / sd_e * np.sqrt(252) if sd_e > 0 else np.nan),
                abs_sharpe=float(y.mean() / y.std() * np.sqrt(252)))


def main():
    cur = pd.read_pickle(OUT / "strategy_comparison_curves.pkl")
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl")
    b = load_bars("QQQ")
    intra = (b.day_close / b.day_open - 1).reindex(bh.index).fillna(0.0)

    rows = []
    for name in cur.columns:
        r = cur[name].dropna()
        rec = dict(版本=name)
        for per, pl in ((DEV, "dev"), (TEST, "test"), (None, "全")):
            x = r if per is None else r.loc[per[0]:per[1]]
            for bench, bn in ((bh, "BH"), (intra, "日内")):
                d = decomp(x, bench)
                for k, v in d.items():
                    if k == "abs_sharpe" and bn == "日内":
                        continue          # 绝对 Sharpe 与基准无关，只存一次
                    rec[f"{pl}_{bn}_{k}" if k != "abs_sharpe" else f"{pl}_abs_sharpe"] = v
        rows.append(rec)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "alpha_beta_decomp.csv", index=False)

    SHORT = {
        "多空 vwap+尾1.0 全周": "①SOTA基线",
        "多空 vwap+尾1.0 全周+空头止盈": "②SOTA+止盈",
        "多空 vwap+尾1.0 无周一": "③SOTA+无周一(实盘)",
        "多空 vwap+尾1.0 无周一+空头止盈": "④SOTA+无周一+止盈",
        "多空 band+尾0.5 全周": "⑤band",
        "多空 band+尾0.5 全周+空头止盈": "⑥band+止盈",
        "多空 band+尾0.5 无周一": "⑦band+无周一",
        "多空 band+尾0.5 无周一+空头止盈": "⑧band+无周一+止盈",
        "只做多 vwap+尾1.0 全周": "⑨只多·vwap",
        "只做多 vwap+尾1.0 无周一": "⑩只多·vwap无周一",
        "只做多 band+尾0.5 全周": "⑪只多·band",
        "只做多 band+尾0.5 无周一": "⑫只多·band无周一",
    }
    R["s"] = R.版本.map(SHORT)
    pd.set_option("display.width", 260)

    def show(title, cols, hdr):
        print("\n" + "=" * 146); print(title); print("=" * 146)
        t = R[["s"] + cols].copy(); t.columns = ["版本"] + hdr
        print(t.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    show("【vs 全日 B&H】β / α（年化）/ α 的 HAC t 值 / alpha Sharpe / 绝对 Sharpe —— test 期",
         ["test_abs_sharpe", "test_BH_beta", "test_BH_alpha_ann", "test_BH_t_alpha",
          "test_BH_alpha_sharpe"],
         ["绝对SR", "β(BH)", "α年化", "α的t(HAC)", "alpha SR"])

    show("【vs 全日 B&H】dev 期",
         ["dev_abs_sharpe", "dev_BH_beta", "dev_BH_alpha_ann", "dev_BH_t_alpha",
          "dev_BH_alpha_sharpe"],
         ["绝对SR", "β(BH)", "α年化", "α的t(HAC)", "alpha SR"])

    show("【vs 日内 open→close】—— 回答『是不是只是在吃日内市场收益』（test 期）",
         ["test_日内_beta", "test_日内_alpha_ann", "test_日内_t_alpha",
          "test_日内_alpha_sharpe"],
         ["β(日内)", "α年化", "α的t(HAC)", "alpha SR"])

    show("【全样本 2019-2026】vs 全日 B&H",
         ["全_abs_sharpe", "全_BH_beta", "全_BH_alpha_ann", "全_BH_t_alpha",
          "全_BH_alpha_sharpe"],
         ["绝对SR", "β(BH)", "α年化", "α的t(HAC)", "alpha SR"])

    print("\n" + "=" * 146)
    print("绝对 Sharpe 与 alpha Sharpe 的差 = β 贡献的部分（test，vs 全日 B&H）")
    d = (R.test_abs_sharpe - R.test_BH_alpha_sharpe)
    for s, v, bt in zip(R.s, d, R.test_BH_beta):
        print(f"  {s:22s} 绝对−alpha = {v:+.3f}   (β={bt:+.3f})")
    print(f"\n表 -> {OUT}/alpha_beta_decomp.csv")


if __name__ == "__main__":
    main()
