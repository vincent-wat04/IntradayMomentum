"""用 mdrec 真实逐笔/盘口(2026-10-20 ~ 10-30,5 个完整交易日 + 1 个半天)验证此前研究用的 1 分钟代理与成本假设。
不做收益可预测性检验(样本太短),只做"代理 vs 真值"与"成本假设 vs 实测"。
  V1 BVC(Easley–López de Prado–O'Hara)买入量份额 vs 真实主动买卖(只用 AUTO_MATCH + ODD_LOT;CONTINGENT 等大宗/条件单剔除)
  V2 1 分钟 OHLC 的 Corwin–Schultz 价差 vs 实测 L1 价差
  V3 回测成本假设:半价差 $0.005/股 + 佣金 $0.0045/股(TQQQ)vs 实测 TQQQ 半价差
  V4 Cont–Kukanov–Stoikov OFI 对 1 分钟中间价变动的解释力(R²);真实成交失衡对 |收益| 的关系(E1 入场带的"还手"假设)
python3 hf_validate.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_mdrec as H
from etft.intraday.bars import load_bars

REG = ("AUTO_MATCH", "ODD_LOT")


def regular_flow(code):
    df = H.read_all(code, "ticker"); df["ts"] = pd.to_datetime(df["time"], format="ISO8601"); df = df.drop_duplicates("sequence")
    for c in ("price", "volume"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["price", "volume"]); t = df.ts.dt.time
    df = df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time()) & df["type"].isin(REG)]
    df["day"] = df.ts.dt.strftime("%Y-%m-%d"); df["m"] = H.minute_key(df.ts)
    df["b"] = df.volume * (df.direction == "BUY"); df["s"] = df.volume * (df.direction == "SELL")
    g = df.groupby(["day", "m"]); o = pd.DataFrame({"buy": g.b.sum(), "sell": g.s.sum(), "vol_reg": g.volume.sum(), "vwap": (df.price * df.volume).groupby([df.day, df.m]).sum() / g.volume.sum()})
    return o


def main():
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:9.4f}")
    HF = pd.read_pickle(H.OUT); res = {}
    for code, sym in (("US.QQQ", "QQQ"), ("US.TQQQ", None)):
        fl = regular_flow(code)
        rows = []
        for (d, c), x in HF.items():
            if c != code: continue
            y = x.join(fl.loc[d], how="left") if d in fl.index.get_level_values(0) else x
            y["day"] = d; rows.append(y)
        res[code] = pd.concat(rows)
    # ---- 1 分钟 bar(QQQ:repo 的 moomoo 分钟线只到 10-01,用 mdrec 自己重建 OHLC)
    def bars_from_ticks(code):
        df = H.read_all(code, "ticker"); df["ts"] = pd.to_datetime(df["time"], format="ISO8601"); df = df.drop_duplicates("sequence")
        for c in ("price", "volume"): df[c] = pd.to_numeric(df[c], errors="coerce")
        t = df.ts.dt.time; df = df.dropna(subset=["price"])[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
        df = df[df["type"].isin(REG)].sort_values("ts"); df["day"] = df.ts.dt.strftime("%Y-%m-%d"); df["m"] = H.minute_key(df.ts)
        g = df.groupby(["day", "m"]).price; return pd.DataFrame({"o": g.first(), "h": g.max(), "l": g.min(), "c": g.last(), "v": df.groupby(["day", "m"]).volume.sum()})
    out = []
    for code in ("US.QQQ", "US.TQQQ"):
        B = bars_from_ticks(code); X = res[code].reset_index().rename(columns={"index": "m"}).set_index(["day", "m"]).join(B, how="inner")
        X["r"] = X.groupby(level=0).c.apply(lambda s: np.log(s).diff()).droplevel(0)
        X["dmid"] = X.groupby(level=0).mid_last.apply(lambda s: np.log(s).diff()).droplevel(0)
        # V1 BVC:买入份额 = Φ(r / σ),σ = 当日此前的 1 分钟收益标准差(只用过去,当日前 20 分钟用全日以前数据不足 → 跳过)
        sig = X.groupby(level=0).r.apply(lambda s: s.shift(1).expanding(20).std()).droplevel(0)
        X["bvc"] = norm.cdf(X.r / sig)
        X["true_share"] = X.buy / (X.buy + X.sell)
        m = X[["bvc", "true_share"]].notna().all(1) & ((X.buy + X.sell) > 0)
        v1 = dict(代码=code, 分钟=int(m.sum()), V1_BVC与真实买入份额_Pearson=np.corrcoef(X.bvc[m], X.true_share[m])[0, 1], V1_Spearman=spearmanr(X.bvc[m], X.true_share[m])[0],
                  V1_真实净流与收益符号一致率=float((np.sign(X.buy - X.sell)[m] == np.sign(X.r)[m]).mean()))
        # V2 Corwin–Schultz(相邻两根 bar)
        hl = np.log(X.h / X.l) ** 2; beta = hl + hl.groupby(level=0).shift(1)
        hh = pd.concat([X.h, X.h.groupby(level=0).shift(1)], axis=1).max(1); ll = pd.concat([X.l, X.l.groupby(level=0).shift(1)], axis=1).min(1)
        gam = np.log(hh / ll) ** 2; k = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gam / k); cs = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0) * 1e4
        mm = cs.notna() & X.spread_bp.notna()
        v1.update(V2_CS价差中位bp=float(cs[mm].median()), V2_实测L1价差中位bp=float(X.spread_bp[mm].median()), V2_逐分钟相关=float(spearmanr(cs[mm], X.spread_bp[mm])[0]),
                  V2_日均相关=float(np.corrcoef(cs[mm].groupby(level=0).mean(), X.spread_bp[mm].groupby(level=0).mean())[0, 1]))
        # V4 OFI → Δmid;成交失衡 → |r|
        m4 = X[["ofi", "dmid"]].notna().all(1) & np.isfinite(X.dmid)
        dep = X.depth1.groupby(level=0).transform("mean"); z = X.ofi / dep
        v1.update(V4_OFI解释Δmid_R2=float(np.corrcoef(z[m4], X.dmid[m4])[0, 1] ** 2), V4_净成交解释Δmid_R2=float(np.corrcoef(((X.buy - X.sell) / X.vol_reg)[m4].fillna(0), X.dmid[m4])[0, 1] ** 2))
        imb = ((X.buy - X.sell).abs() / (X.buy + X.sell)); m5 = imb.notna() & X.r.notna()
        q = pd.qcut(imb[m5].rank(method="first"), 5, labels=False)
        v1.update(V4_失衡五分位的平均绝对收益bp="/".join(f"{x:.1f}" for x in (X.r[m5].abs().groupby(q.to_numpy()).mean() * 1e4)))
        out.append(v1); res[code] = X
    T = pd.DataFrame(out).set_index("代码").T; print("=== 代理 vs 真值(mdrec,常规时段每分钟)==="); print(T.to_string())
    # V3 成本
    Q = res["US.TQQQ"]; px = Q.c; half = Q.spread_bp / 2 * px / 1e4
    print(f"\n=== V3 TQQQ 执行成本(实测)===\n  价格中位 ${px.median():.2f};实测半价差中位 ${half.median():.4f}/股(p90 ${half.quantile(.9):.4f});按 bp 计 {Q.spread_bp.median()/2:.2f}bp")
    for t_ in ("10:00", "10:30", "12:00", "15:30", "15:40"):
        x = Q.xs(t_, level=1); print(f"  {t_}:半价差中位 ${(x.spread_bp/2*x.c/1e4).median():.4f}/股,L1 深度中位 {x.depth1.median():.0f} 股")
    print(f"  回测假设:半价差 $0.0050/股 + 佣金 $0.0045/股 → 单边 ${0.0095:.4f}/股;实测半价差/假设 = {half.median()/0.005:.2f}×")
    pd.to_pickle(res, Path(os.environ.get("ETFT_DATA", "../data")) / "_hf_validate.pkl")


if __name__ == "__main__":
    main()
