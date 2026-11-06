"""多源新闻 N 组 → 次日日内波动 / 低波标签(docs/PROPOSAL_news_multisource.md,第一阶段:非 LLM 特征)。
python3 lowvol_round4_news_study.py"""
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
import lowvol_avoidance_study as L
import news_ms as N
from event_calendar import event_frame
from etft.intraday.bars import load_bars

OUT, TAB = L.OUT, L.TAB
OOS_END = "2023"
L.OOS_YEARS = range(2021, 2024)          # 新闻覆盖到 2023-12(无 2024+ 的 W 特征可预测)


def selftest(items, days):
    d0 = pd.Timestamp("2022-06-15"); open0 = pd.Timestamp(f"{d0.date()} 09:30", tz="America/New_York").tz_convert("UTC")
    F0 = N.daily_features(items, days); Fl0 = N.daily_features(items, days, leak=True)
    bad = items.copy(); rng = np.random.default_rng(3)
    m_min = (bad.prec != "day") & (bad.ts > open0); m_day = (bad.prec == "day") & (bad.dt > d0)
    shift = pd.to_timedelta(rng.integers(200, 600, len(bad)), unit="D")
    bad.loc[m_min, "ts"] = bad.loc[m_min, "ts"] + shift[m_min.to_numpy()]
    bad.loc[m_day, "dt"] = bad.loc[m_day, "dt"] + shift[m_day.to_numpy()]
    F1 = N.daily_features(bad, days); Fl1 = N.daily_features(bad, days, leak=True)
    sl = slice(None, d0)
    a, b = F0.loc[sl].to_numpy(), F1.loc[sl].to_numpy()
    same_nan = (np.isnan(a) == np.isnan(b)).all(); dmax = float(np.nanmax(np.abs(a - b)))
    al, bl = Fl0.loc[sl].to_numpy(), Fl1.loc[sl].to_numpy(); dleak = float(np.nanmax(np.abs(al - bl)))
    ok = same_nan and dmax == 0.0 and dleak > 0
    print(f"[自检] 把 {d0.date()} 09:30 ET 之后的新闻时间戳整体后移 200~600 天(共 {int(m_min.sum()+m_day.sum())} 条):≤{d0.date()} 特征最大差 {dmax:.2e}(须为 0,NaN 模式一致={same_nan});"
          f"阳性对照(盘中新闻并入当日)最大差 {dleak:.2f}(须>0) → {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    b = load_bars("QQQ"); days = b.days
    items = N.load_items()
    print(f"新闻条目 {len(items)}:", items.groupby("src").size().to_dict())
    if not selftest(items, days): print("自检失败,终止"); return
    W = N.daily_features(items, days)
    evaluate(W, "")


def evaluate(W, tag=""):
    b = load_bars("QQQ"); days = b.days
    X, Y, A, B = L.build(b)
    E = event_frame(days, b.last_idx.to_numpy()); EC = list(E.columns)
    WC = list(W.columns)
    X = pd.concat([X, E, W], axis=1)
    sets = {"A·ridge": (A, "ridge"), "A+B+E·ridge": (A + B + EC, "ridge"), "W·ridge": (WC, "ridge"), "A+W·ridge": (A + WC, "ridge"),
            "A+B+E+W·ridge": (A + B + EC + WC, "ridge"), "A+B+E+W·lgbm": (A + B + EC + WC, "lgbm")}
    pred, base_rv, thr_s = L.walk_forward(X, Y, sets)
    oos = X.index[(X.index.year >= 2021) & (X.index.year <= int(OOS_END)) & X[A + B + EC + WC].notna().all(1) & Y.notna().all(1)]
    dev = oos[oos.year <= 2022]; test = oos[oos.year == 2023]
    print(f"OOS 天数 {len(oos)}(dev 2021–22: {len(dev)};test 2023: {len(test)})")
    ylow = pd.Series((Y.loc[oos, "disp"] <= thr_s.loc[oos]).astype(int).to_numpy(), index=oos)
    sse_b = pd.Series(((Y.loc[oos, "y_rv"] - base_rv.loc[oos]) ** 2).to_numpy(), index=oos)
    pd.set_option("display.width", 230)
    rows = []
    for (nm, key), p in pred.items():
        p = p.loc[oos]
        if key == "rv":
            e = pd.Series(((Y.loc[oos, "y_rv"] - p) ** 2).to_numpy(), index=oos); pt, lo, hi = L.month_boot(oos, lambda ii, e=e: 1 - e.iloc[ii].sum() / sse_b.iloc[ii].sum())
            rows.append(dict(模型=nm, 指标="OOS R² (log RV)", 值=pt, lo=lo, hi=hi))
        else:
            pp = p.to_numpy(); pt, lo, hi = L.month_boot(oos, lambda ii, pp=pp: L.auc(ylow.to_numpy()[ii], pp[ii])); rows.append(dict(模型=nm, 指标="AUC(低波)", 值=pt, lo=lo, hi=hi))
    R1 = pd.DataFrame(rows); print("\n1) 预测力(OOS 2021–2023)"); print(R1.round(4).to_string(index=False))
    inc = []; gate = {}
    for n1, n0 in (("A+B+E+W·ridge", "A+B+E·ridge"), ("A+W·ridge", "A·ridge"), ("A+B+E+W·lgbm", "A+B+E·ridge")):
        e1 = pd.Series(((Y.loc[oos, "y_rv"] - pred[(n1, "rv")].loc[oos]) ** 2).to_numpy(), index=oos); e0 = pd.Series(((Y.loc[oos, "y_rv"] - pred[(n0, "rv")].loc[oos]) ** 2).to_numpy(), index=oos)
        for per, idx in (("全OOS", oos), ("dev", dev), ("test", test)):
            sub = idx
            fn = lambda ii, sub=sub: (e0.loc[sub].iloc[ii].sum() - e1.loc[sub].iloc[ii].sum()) / sse_b.loc[sub].iloc[ii].sum()
            pt, lo, hi = L.month_boot(sub, fn); inc.append(dict(比较=f"{n1} − {n0}", 期间=per, 指标="ΔR²", 值=pt, lo=lo, hi=hi))
            if n1 == "A+B+E+W·ridge": gate[per] = (pt, lo, hi)
        p1, p0 = pred[(n1, "low")].loc[oos].to_numpy(), pred[(n0, "low")].loc[oos].to_numpy(); yl = ylow.to_numpy()
        pt, lo, hi = L.month_boot(oos, lambda ii: L.auc(yl[ii], p1[ii]) - L.auc(yl[ii], p0[ii])); inc.append(dict(比较=f"{n1} − {n0}", 期间="全OOS", 指标="ΔAUC", 值=pt, lo=lo, hi=hi))
    R2 = pd.DataFrame(inc); print("\n2) 增量(主指标:A+B+E+W·ridge − A+B+E·ridge 的 ΔR²)"); print(R2.round(4).to_string(index=False))
    e1 = ((Y.loc[oos, "y_rv"] - pred[("A+B+E+W·ridge", "rv")].loc[oos]) ** 2); e0 = ((Y.loc[oos, "y_rv"] - pred[("A+B+E·ridge", "rv")].loc[oos]) ** 2)
    print("逐年 ΔSSE%(>0 = W 组改善):", {int(y): round(float((e0[oos.year == y].sum() - e1[oos.year == y].sum()) / e0[oos.year == y].sum()), 4) for y in sorted(set(oos.year))})
    # 单变量:W 各特征 vs 『A+B+E 残差』
    res = (Y.y_rv - pred[("A+B+E·ridge", "rv")]).loc[oos]; res_A = (Y.y_rv - pred[("A·ridge", "rv")]).loc[oos]
    U = []
    for c in WC:
        r_ = {"特征": c}
        for per, idx in (("全OOS", oos), ("dev", dev), ("test", test)):
            r_[f"rho_{per}"] = spearmanr(X.loc[idx, c], res.loc[idx])[0]
        r_["rho_全OOS(vs A残差)"] = spearmanr(X.loc[oos, c], res_A)[0]
        n = len(oos); t = r_["rho_全OOS"] * np.sqrt((n - 2) / (1 - r_["rho_全OOS"] ** 2)); from scipy.stats import t as tdist
        r_["p_单"] = 2 * tdist.sf(abs(t), n - 2); r_["同号(dev,test)"] = np.sign(r_["rho_dev"]) == np.sign(r_["rho_test"]); U.append(r_)
    U = pd.DataFrame(U); print("\n3) 单变量 Spearman(特征 vs 残差;Bonferroni 阈值 p<%.4f)" % (0.05 / len(WC))); print(U.round(3).to_string(index=False))
    print("过 Bonferroni 的特征:", U[U.p_单 < 0.05 / len(WC)].特征.tolist())
    R1.to_csv(TAB / f"news_ms{tag}_prediction.csv", index=False); R2.to_csv(TAB / f"news_ms{tag}_incremental.csv", index=False); U.to_csv(TAB / f"news_ms{tag}_univariate.csv", index=False)
    pt, lo, hi = gate["全OOS"]; ok = lo > 0 and gate["dev"][0] > 0 and gate["test"][0] > 0
    print(f"\n闸门(§3):全OOS ΔR² 区间 [{lo:.4f},{hi:.4f}] {'不含' if lo>0 else '含'} 0;dev {gate['dev'][0]:+.4f}、test {gate['test'][0]:+.4f} → {'通过,可进入变现' if ok else '未通过:按预注册不跑变现'}")
    pd.to_pickle(dict(pred=pred, W=W, oos=oos), OUT / f"news_ms{tag}_round4.pkl")


if __name__ == "__main__":
    main()
