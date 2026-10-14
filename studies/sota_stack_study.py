"""避开低振幅日的模型赛马 / 反向仓位倾斜 / Sleeve SOTA 组装 / B&H 腿参数(跨周期)—— docs/PROPOSAL_sota_stack.md(预注册)。
python3 sota_stack_study.py"""
from __future__ import annotations

import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_pred_study as DP
import ffa_vol_study as FV
import lowvol_avoidance_study as L
import macro_events_study as M
import regime_features as RF
import regime_timing_study as R1
import round3_common as C
import trend_rv_sleeve_study as T
from audit import reimpl_noise as RI
from event_calendar import FOMC_DECISION
from etft.intraday.bars import load_bars
from sens_rv_weights import pct_w, w_linear
import paths as _P

OUT, TAB = L.OUT, L.TAB
NPERM = 1000; QS = (0.1, 0.2, 0.3)
SEL0, SEL1 = "2015-01-01", "2018-12-31"


# ============================================================ 工具
def panel(x, name=""):
    x = pd.Series(x).dropna(); d = x[x < 0]; eq = (1 + x).cumprod(); mdd = float((eq / eq.cummax() - 1).min()); yrs = len(x) / 252
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else np.nan; dn = np.sqrt((np.minimum(x, 0) ** 2).mean()); prof = x[x > 0].sum()
    return {"方案": name, "均值bp": 1e4 * x.mean(), "中位bp": 1e4 * x.median(), "年化波动": x.std() * np.sqrt(252), "Sharpe": x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan,
            "Sortino": x.mean() / dn * np.sqrt(252) if dn > 0 else np.nan, "CAGR": cagr, "MDD": mdd, "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "偏度": skew(x), "峰度": kurtosis(x),
            "CVaR5%bp": 1e4 * x[x <= x.quantile(0.05)].mean(), "前5%日占利润": x.nlargest(max(int(.05 * len(x)), 1)).sum() / prof if prof > 0 else np.nan, "最差日bp": 1e4 * x.min()}


def roll_rank(F, w=504, minp=126):
    v = F.to_numpy(float); out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if not np.isfinite(v[i]): continue
        h = v[max(0, i - w):i]; h = h[np.isfinite(h)]
        if len(h) >= minp: out[i] = np.mean(h < v[i])
    return pd.Series(out, index=F.index)


def auc(y, s):
    m = np.isfinite(s) & np.isfinite(y); return L.auc(y[m].astype(int), s[m])


# ============================================================ A. 候选预测
def ml_forecasts(R, r, q, vix, extra=None, start_year=2005, min_train=750):
    """岭回归(逐年扩展窗口)预测 ln R_t;特征只用 t−1 收盘前(+ 可选的开盘/10:00 信息)。返回水平预测(对齐到目标日 t)。"""
    lR = np.log(R); X = pd.DataFrame({"lR1": lR.shift(1), "lR5": np.log(R.rolling(5).mean()).shift(1), "lR22": np.log(R.rolling(22).mean()).shift(1),
                                      "lvix": np.log(vix).shift(1), "dlvix": np.log(vix).diff().shift(1), "absr": r.abs().shift(1), "neg": (r < 0).astype(float).shift(1),
                                      **{f"d{k}": (R.index.dayofweek == k).astype(float) for k in (1, 2, 3, 4)}}, index=R.index)
    if extra is not None: X = X.join(extra)
    y = lR; out = pd.Series(np.nan, index=R.index)
    for yr in range(start_year, 2027):
        tr = (X.index < f"{yr}-01-01") & X.notna().all(1).to_numpy() & y.notna().to_numpy(); te = (X.index.year == yr) & X.notna().all(1).to_numpy()
        if tr.sum() < min_train or te.sum() == 0: continue
        mu, sd = X[tr].mean(), X[tr].std().replace(0, 1); A = ((X[tr] - mu) / sd).to_numpy(); B = ((X[te] - mu) / sd).to_numpy()
        m = Ridge(alpha=1.0).fit(A, y[tr]); res = y[tr] - m.predict(A); out[te] = np.exp(m.predict(B)) * np.mean(np.exp(res))
    return out


def build_candidates(b):
    S = pd.read_pickle(OUT / "ffa_forecasts.pkl"); FC, dates, R = S["FC"], S["dates"], S["R"]
    q, _, r = FV.load(); vix = pd.read_parquet(_P.RAW / "cboe" / "VIX.parquet")["close"]; vix.index = pd.to_datetime(vix.index).tz_localize(None); vix = vix[~vix.index.duplicated()].reindex(dates).ffill()
    cand = {k: pd.Series(v[1], index=dates) for k, v in FC.items()}
    base9 = [k for k in cand if k != "FFA-MAX"]
    cand["COMB"] = np.exp(pd.concat([np.log(cand[k]) for k in base9], axis=1).mean(axis=1, skipna=False))
    cand["ML-盘前"] = ml_forecasts(R, r, q, vix)
    adj = q.close / q.close_raw; gap = np.abs(np.log((q.open * adj) / q.close.shift(1))).rename("gap")      # 调整后开盘 / 调整后前收(开盘时已知)
    cand["ML-开盘"] = ml_forecasts(R, r, q, vix, extra=pd.DataFrame({"gap": gap}))
    # ML-10:00(分钟线 2019+,不参与选择)
    days = b.days; c = b.c.to_numpy(); h = b.h.to_numpy(); l = b.l.to_numpy(); O = b.day_open.to_numpy()
    hi30 = np.maximum(h[:, :30].max(1), O); lo30 = np.minimum(l[:, :30].min(1), O); p30 = pd.Series((np.log(hi30) - np.log(lo30)) ** 2 / (4 * np.log(2)) * 100, index=days)
    from etft.intraday.strategy import rel_volume
    rv30 = pd.Series(np.log(rel_volume(b, 14, "cum")["10:00"].to_numpy()), index=days)
    ex = pd.DataFrame({"gap": gap, "lp30": np.log(p30.where(p30 > 0)).reindex(dates), "rv30": rv30.reindex(dates)})
    cand["ML-10:00"] = ml_forecasts(R, r, q, vix, extra=ex, start_year=2020, min_train=250)
    return cand, R, dates


# ============================================================ Sleeve 组件
def fomc_late(b, scale):
    """FOMC 日只在 ≥14:30 检查点开仓的 Sleeve(只模拟 FOMC 日)。"""
    days = b.days; fs = set(d for d in FOMC_DECISION if d in set(days)); nonf = [d for d in days if d not in fs]; rs = []
    for off in range(30):
        checks = [t for t in DP.SA.chk(off) if t >= "14:30"]
        rr, F = RI.simulate(b, checks=checks, sizing="voltarget", max_lev=4.0, reentry="hold", half_spread=0.005, commission=0.0045, skip_days=nonf, **DP.SA.BASE)
        if len(F):
            F = F.copy(); F["day"] = pd.to_datetime(F.day); c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0); rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
        rs.append(rr)
    return pd.concat(rs, axis=1).mean(axis=1), pd.Index(sorted(fs))


def cpi_overlay(b, exclude_monday=True):
    """CPI 缺口回补 overlay。exclude_monday=True 与 Sleeve 的 skip_dows=(0,) 保持一致:
    整个 Sleeve(含 overlay)周一不交易。样本内 2019–2025 的 CPI 发布无一在周一,所以这是结构性约束而非改变结果。"""
    days = b.days; EV, _, _ = M.load_events(days); cpi = EV["CPI"]; sg = np.sign(cpi.sf); sg = sg[sg != 0]
    oc = pd.Series(b.day_close.to_numpy() / b.day_open.to_numpy() - 1, index=days); ov = pd.Series(0.0, index=days); mm = sg.index.intersection(days); ov.loc[mm] = sg.loc[mm] * oc.loc[mm] - 2e-4
    if exclude_monday: ov[ov.index.dayofweek == 0] = 0.0
    return ov


# ============================================================ D. B&H 腿
def bh_weights(px, r, F_by, A2C):
    p = pct_w(r.rolling(21).std(), 504); W = {}
    pos = {"静态": pd.Series(1.0, index=px.index)}
    for lo, hi in ((0.5, 1.5), (0.0, 1.0), (0.0, 1.5), (0.0, 2.0)): pos[f"RV21[{lo:g},{hi:g}]"] = w_linear(p, lo, hi).shift(1).fillna(1.0)
    for nm, F in F_by.items():
        F = F.reindex(px.index); sv = np.sqrt(F); c_ = sv.rolling(252, min_periods=126).median().shift(1)
        for cap in (1.5, 2.0): pos[f"VT[{nm},cap{cap:g}]"] = np.minimum(cap, c_ / sv).fillna(1.0)
    trend = {"无趋势": pd.Series(1.0, index=px.index)}
    for kind, n in (("SMA", 200), ("MOM", 252)):
        dn = T.trend_down(px, kind, n).astype(float)
        for e in (0.5, 0.0): trend[f"{kind}{n}|e={e:g}"] = (1.0 - dn * (1.0 - e)).shift(1).fillna(1.0)
    trend["信用A2_C|e=0.5"] = A2C.reindex(px.index).fillna(1.0)
    for pn, w in pos.items():
        for tn, t_ in trend.items(): W[(pn, tn)] = (w * t_)
    return W


def bh_returns(w, r, rf, interest):
    turn = w.diff().abs().fillna(0.0); x = w * r - 2e-4 * turn
    if interest: x = w * (r - rf) - 2e-4 * turn                     # 超额收益口径(w>1 付融资,w<1 现金得息)
    return x


def main():
    t0 = time.time(); pd.set_option("display.width", 280, "display.max_rows", 500, "display.float_format", lambda x: f"{x:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year; td = pd.Series(days.dayofweek != 0, index=days); sel = td.to_numpy() & (yrs >= 2019)
    base = pd.read_pickle(OUT / "sleeve_anatomy_curve.pkl"); base.index = pd.to_datetime(base.index); base = base.reindex(days).fillna(0.0)
    sd14 = RI.daily_sigma(b.day_close, b.prev_close_adj, 14).reindex(days)
    # ======================================================= A. 赛马
    cand, R, dates = build_candidates(b); print(f"候选 {list(cand)}({time.time()-t0:.0f}s)")
    thr = R.rolling(504, min_periods=252).quantile(0.2).shift(1); low = (R <= thr).astype(float).where(thr.notna())
    rows = []
    for nm, F in cand.items():
        for pn, (a_, z_) in (("选择期 2015–18", (SEL0, SEL1)), ("2019–26", ("2019-01-01", "2026-10-09"))):
            m = (dates >= a_) & (dates <= z_) & F.notna().to_numpy() & low.notna().to_numpy()
            Rm, Fm = R[m].to_numpy(), F[m].to_numpy(); ql = np.mean(Rm / Fm - np.log(Rm / Fm) - 1)
            rows.append(dict(模型=nm, 期间=pn, n=int(m.sum()), AUC低振幅=auc(low[m].to_numpy(), -Fm) if m.sum() > 100 else np.nan, QLIKE=ql, MAE=np.mean(np.abs(Rm - Fm))))
    RA = pd.DataFrame(rows); pv = RA.pivot_table(index="模型", columns="期间", values=["AUC低振幅", "QLIKE", "MAE"], sort=False)
    print("\nA1) 低振幅日识别力(AUC;标签 = 振幅 ≤ 过去 504 日 20% 分位)"); print(pv.round(4).to_string()); RA.to_csv(TAB / "sota_race_stat.csv", index=False)
    elig = RA[(RA.期间 == "选择期 2015–18") & (RA.模型 != "ML-10:00")].sort_values(["AUC低振幅", "QLIKE"], ascending=[False, True]).reset_index(drop=True)
    top = elig.iloc[0]; tie = elig[(top.AUC低振幅 - elig.AUC低振幅) < 0.005]; winner = tie.sort_values("QLIKE").iloc[0].模型
    print(f"\n预注册选择(2015–18 的 AUC,并列时取 QLIKE 低):胜者 = {winner}(AUC {elig.set_index('模型').loc[winner,'AUC低振幅']:.4f});并列组 {list(tie.模型)}")
    ranks = {nm: roll_rank(F.reindex(days) if nm == "ML-10:00" else F).reindex(days) for nm, F in cand.items()}
    # 经济:13+1 候选 × 3 q
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    port = lambda sl: T.build(bh, sl.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px_all, p_all)[0]
    base_p = port(base); idx = np.where(sel)[0]; rg = np.random.default_rng(3); perms = [rg.permutation(len(idx)) for _ in range(NPERM)]; bsr = L.sr(base.iloc[idx])
    rows, nulls, obs = [], {}, {}
    def per_sr(x, k): mk = sel & ((yrs <= 2022) if k == "dev" else (yrs >= 2023) if k == "test" else (yrs >= 2019)); return L.sr(x[mk])
    def pper(x, k): xi = x[x.index >= "2019-01-01"]; return L.sr(xi[xi.index.year <= 2022] if k == "dev" else xi[xi.index.year >= 2023] if k == "test" else xi)
    for nm in cand:
        for qq in QS:
            mask = (ranks[nm] <= qq).fillna(False); s = (~mask).astype(float); x = base * s; xp = port(x); sv = s.to_numpy()
            o = L.sr(base.iloc[idx] * sv[idx]) - bsr; obs[(nm, qq)] = o
            nulls[(nm, qq)] = np.array([L.sr(pd.Series(base.to_numpy()[idx] * sv[idx][pi])) for pi in perms]) - bsr
            rows.append(dict(模型=nm, q=qq, 跳过比例=float(mask[sel].mean()), **{f"Sleeve ΔSR {k}": per_sr(x, k) - per_sr(base, k) for k in ("dev", "test", "全")},
                             **{f"组合 ΔSR {k}": pper(xp, k) - pper(base_p, k) for k in ("dev", "test", "全")}, 组合年化bp=1e4 * 252 * (xp[xp.index >= "2019"].mean() - base_p[base_p.index >= "2019"].mean()),
                             被跳过日Sleeve均bp=1e4 * float(base[sel & mask.to_numpy()].mean()), p单=float(np.mean(nulls[(nm, qq)] >= o))))
    keys = [k for k in nulls if k[0] != "ML-10:00"]; fam = np.max(np.vstack([nulls[k] for k in keys]), axis=0)
    EA = pd.DataFrame(rows); EA["p族内(39)"] = [float(np.mean(fam >= obs[(m_, q_)])) if m_ != "ML-10:00" else np.nan for m_, q_ in zip(EA.模型, EA.q)]
    print("\nA2) 跳过预测最低振幅日的经济效果(2019–2026;Sleeve 层与组合层)"); print(EA.to_string(index=False)); EA.to_csv(TAB / "sota_race_econ.csv", index=False)
    ew = EA[EA.模型 == winner].set_index("q"); q_star = float(ew["Sleeve ΔSR dev"].idxmax())
    print(f"\nSOTA 跳过参数:模型 {winner},q 在 dev 选 = {q_star}(dev Sleeve ΔSR {ew.loc[q_star,'Sleeve ΔSR dev']:+.3f};test {ew.loc[q_star,'Sleeve ΔSR test']:+.3f})")
    # ======================================================= B. 反向仓位倾斜
    rows, nl, ob = [], {}, {}; shifts = rg.integers(60, len(idx) - 60, NPERM)
    tilts = {}
    for nm in (winner, "FFA-P"):
        F = cand[nm].reindex(days)
        for rk_n, rk in (("绝对", ranks[nm]), ("相对σ₁₄²", roll_rank((F / (100 * sd14 ** 2)).where(F.notna())).reindex(days))):
            for (a_, b_), tn in (((0.5, 1.0), "强"), ((0.75, 0.5), "温和")):
                expo0 = np.minimum(4.0, 0.02 / sd14); g = (a_ + b_ * rk).fillna(1.0); ratio = (np.minimum(4.0, expo0 * g) / expo0).fillna(1.0)
                x = base * ratio; xp = port(x); key = f"{nm}|{rk_n}|{tn}"; tilts[key] = ratio; rv_ = ratio.to_numpy()[idx]; bv = base.to_numpy()[idx]
                ob[key] = L.sr(pd.Series(bv * rv_)) - bsr; nl[key] = np.array([L.sr(pd.Series(bv * np.roll(rv_, s))) for s in shifts]) - bsr
                rows.append(dict(倾斜=key, 平均倍数=float(ratio[sel].mean()), **{f"Sleeve ΔSR {k}": per_sr(x, k) - per_sr(base, k) for k in ("dev", "test", "全")},
                                 **{f"组合 ΔSR {k}": pper(xp, k) - pper(base_p, k) for k in ("dev", "test", "全")}, Sleeve均值变化bp=1e4 * float(x[sel].mean() - base[sel].mean())))
    famb = np.max(np.vstack(list(nl.values())), axis=0); EB = pd.DataFrame(rows); EB["p族内(8)"] = [float(np.mean(famb >= ob[k])) for k in EB.倾斜]
    print("\nB) 反向仓位倾斜:expo = min(4, expo_σ₁₄ × (a + b·rank(预测振幅)))"); print(EB.to_string(index=False)); EB.to_csv(TAB / "sota_tilt.csv", index=False)
    # ======================================================= C. SOTA 组装
    scale = DP.tqqq_scale(b); fl, fset = fomc_late(b, scale); ov = cpi_overlay(b)
    S0 = base.copy(); S1 = S0.copy(); S1.loc[fset] = fl.loc[fset]
    skip = (ranks[winner] <= q_star).fillna(False); S2 = S1 * (~skip).astype(float); S3 = S2 + ov; S3h = S2 + 0.5 * ov
    best_tilt = EB.sort_values("Sleeve ΔSR dev", ascending=False).iloc[0].倾斜; S3t = S2 * tilts[best_tilt] + ov
    VER = {"S0 实盘": S0, "S1 +FOMC≥14:30": S1, f"S2 +跳过({winner},q={q_star:g})": S2, "S3 SOTA(+CPI 回补 1×)": S3, "S3' SOTA(CPI 回补 0.5×)": S3h, f"S3t SOTA+倾斜({best_tilt})": S3t}
    rows = []
    for nm, s in VER.items():
        xs = s[td & (yrs >= 2019)]; xp = port(s); xp = xp[xp.index >= "2019-01-01"]; d, dist = T.boot_dsr(xp, base_p[base_p.index >= "2019-01-01"], nb=1500, L=10, seed=0)
        rows.append(dict(版本=nm, **{k: v for k, v in panel(xs).items() if k != "方案"}, Sleeve_dev=per_sr(s, "dev"), Sleeve_test=per_sr(s, "test"), 组合Sharpe=L.sr(xp), 组合ΔSR=d, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5),
                         组合CAGR=panel(xp)["CAGR"], 组合MDD=panel(xp)["MDD"]))
    EC = pd.DataFrame(rows); print("\nC) Sleeve SOTA 组装(Sleeve 层面板为 2019–2026 周二至周五;组合 = L1 RV[0.5,1.5] + Sleeve)"); print(EC.to_string(index=False)); EC.to_csv(TAB / "sota_sleeve_versions.csv", index=False)
    print("   分年 Sleeve 收益(%):"); print(pd.DataFrame({k: (1 + v[td]).groupby(v[td].index.year).prod() - 1 for k, v in VER.items()}).mul(100).round(1).loc[2019:].to_string())
    pd.to_pickle(dict(VER=VER, winner=winner, q=q_star, best_tilt=best_tilt), OUT / "sota_sleeve.pkl")
    # ======================================================= D. B&H 腿
    P = RF.load_P(); px = px_all.copy(); r = px.pct_change().fillna(0.0); rf = (P["^IRX"].reindex(px.index).ffill() / 100 / 252).fillna(0.0)
    A2C = pd.read_pickle(OUT / "regime_timing_r2.pkl")["E2"]["A2_C|e0.5|alone"]
    F_by = {winner: cand[winner], "FFA-SD": cand["FFA-SD"], "HAR": cand["HAR"], "成分CARR": cand["成分CARR"]}
    W = bh_weights(px, r, F_by, A2C)
    PER = {"全 2004-06~2026": ("2004-06-01", "2026-10-09"), "2004-06~2009": ("2004-06-01", "2009-12-31"), "2010~2018": ("2010-01-01", "2018-12-31"), "2015~2026": ("2015-01-01", "2026-10-09"), "2019~2026": ("2019-01-01", "2026-10-09")}
    BEAR = {"GFC": ("2007-10-01", "2009-03-09"), "2018Q4": ("2018-10-01", "2018-12-24"), "COVID": ("2020-02-19", "2020-04-30"), "2022": ("2022-01-01", "2022-12-31")}
    rows = []
    for key, w in W.items():
        for interest in (False, True):
            x = bh_returns(w, r, rf, interest)
            for pn, (a_, z_) in PER.items():
                m = (x.index >= a_) & (x.index <= z_); xx = x[m]; ww = w[m]; ctrl = float(ww.mean()) * (r[m] - (rf[m] if interest else 0))
                row = dict(仓位=key[0], 趋势=key[1], 口径="含利息(超额)" if interest else "原口径", 期间=pn, 平均权重=float(ww.mean()), 年换手=float(ww.diff().abs().sum() / (m.sum() / 252)),
                           Sharpe=L.sr(xx), CAGR=R1.cagr(xx + (rf[m] if interest else 0)), 年化波动=float(xx.std() * np.sqrt(252)), MDD=R1.mdd(xx + (rf[m] if interest else 0)),
                           Sortino=panel(xx)["Sortino"], dSR_vs同暴露静态=L.sr(xx) - L.sr(ctrl))
                row["Calmar"] = row["CAGR"] / abs(row["MDD"]) if row["MDD"] < 0 else np.nan
                rows.append(row)
            if not interest:
                for bn, (a_, z_) in BEAR.items():
                    m = (x.index >= a_) & (x.index <= z_); rows.append(dict(仓位=key[0], 趋势=key[1], 口径="熊市区间收益", 期间=bn, CAGR=float((1 + x[m]).prod() - 1), 平均权重=float(w[m].mean())))
    BD = pd.DataFrame(rows); BD.to_csv(TAB / "sota_bh_leg.csv", index=False)
    print(f"\nD) B&H 腿:{len(W)} 个配置(仓位 × 趋势)× 2 口径 × {len(PER)} 周期 + 熊市;成本 2bp/边")
    main_ = BD[(BD.口径 == "原口径")]
    pvS = main_.pivot_table(index=["仓位", "趋势"], columns="期间", values="Sharpe", sort=False)[list(PER)]
    pvI = BD[BD.口径 == "含利息(超额)"].pivot_table(index=["仓位", "趋势"], columns="期间", values="Sharpe", sort=False)[list(PER)]
    pvM = main_.pivot_table(index=["仓位", "趋势"], columns="期间", values="MDD", sort=False)[list(PER)]
    pvC = main_.pivot_table(index=["仓位", "趋势"], columns="期间", values="CAGR", sort=False)[list(PER)]
    beat = BD[BD.口径 == "熊市区间收益"].pivot_table(index=["仓位", "趋势"], columns="期间", values="CAGR", sort=False)[list(BEAR)]
    full = main_[main_.期间 == "全 2004-06~2026"].set_index(["仓位", "趋势"])[["平均权重", "年换手", "Sharpe", "CAGR", "MDD", "Calmar", "Sortino", "dSR_vs同暴露静态"]]
    full["Sharpe_含利息"] = BD[(BD.口径 == "含利息(超额)") & (BD.期间 == "全 2004-06~2026")].set_index(["仓位", "趋势"]).Sharpe
    full["正周期数(vs静态同趋势? no: vs 静态无趋势)"] = (pvS.sub(pvS.loc[("静态", "无趋势")], axis=1) > 0).sum(axis=1)
    print("\nD1) 全期(2004-06~2026,原口径)按 Sharpe 排序 —— 前 25 与关键对照"); 
    keyrows = [("静态", "无趋势"), ("RV21[0.5,1.5]", "无趋势"), ("静态", "SMA200|e=0.5"), ("RV21[0.5,1.5]", "SMA200|e=0.5")]
    print(pd.concat([full.sort_values("Sharpe", ascending=False).head(25), full.loc[[k for k in keyrows if k in full.index]]]).drop_duplicates().to_string())
    print("\nD2) 分周期 Sharpe(原口径)——前 25(按全期)"); order = full.sort_values("Sharpe", ascending=False).index[:25].tolist() + [k for k in keyrows if k not in full.sort_values("Sharpe", ascending=False).index[:25]]
    print(pvS.loc[order].to_string()); print("\n    分周期 Sharpe(含利息超额口径)"); print(pvI.loc[order].to_string())
    print("\n    分周期 MDD"); print(pvM.loc[order].to_string()); print("\n    分周期 CAGR"); print(pvC.loc[order].to_string()); print("\n    熊市区间收益"); print(beat.loc[order].mul(100).round(1).to_string())
    # 族内置换(全期,原口径,相对同暴露静态的 ΔSR)
    fullm = (px.index >= "2004-06-01"); rv_ = r[fullm].to_numpy(); n = len(rv_); sh = rg.integers(252, n - 252, 500); nulls = []; obsd = {}
    for key, w in W.items():
        ev = w[fullm].to_numpy(); ctrl = L.sr(pd.Series(ev.mean() * rv_)); obsd[key] = L.sr(pd.Series(ev * rv_ - 2e-4 * np.abs(np.diff(ev, prepend=ev[0])))) - ctrl
        nulls.append(np.array([L.sr(pd.Series(np.roll(ev, s) * rv_ - 2e-4 * np.abs(np.diff(np.roll(ev, s), prepend=np.roll(ev, s)[0])))) - ctrl for s in sh]))
    famd = np.max(np.vstack(nulls), axis=0); pd_ = {k: float(np.mean(famd >= v)) for k, v in obsd.items()}
    print(f"\n    族内圆周平移置换(全期,{len(W)} 个配置):p 最小 6 个", {f"{k[0]}+{k[1]}": round(v, 3) for k, v in sorted(pd_.items(), key=lambda kv: kv[1])[:6]})
    # 与 RV21[0.5,1.5] 的差(全期,块自助)
    ref = bh_returns(W[("RV21[0.5,1.5]", "无趋势")], r, rf, False)[fullm]; rows = []
    for key in order[:12]:
        x = bh_returns(W[key], r, rf, False)[fullm]; d, dist = T.boot_dsr(x, ref, nb=1000, L=40, seed=0); rows.append(dict(配置=f"{key[0]}+{key[1]}", ΔSR_vs_RV21=d, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5)))
    print("\n    相对现用 RV21[0.5,1.5]+无趋势 的 ΔSR(全期,块自助 L=40)"); print(pd.DataFrame(rows).to_string(index=False))
    # ======================================================= D3. 2019+ 叠加 Sleeve
    s_live = base.reindex(px.index).fillna(0.0); s_sota = S3.reindex(px.index).fillna(0.0)
    rows = []
    cand_keys = list(dict.fromkeys(order[:12] + keyrows + [("静态", "MOM252|e=0.5"), ("静态", "信用A2_C|e=0.5"), ("RV21[0,1]", "无趋势"), ("RV21[0,2]", "无趋势")]))
    for key in cand_keys:
        if key not in W: continue
        w = W[key]
        for sn, s in (("无 Sleeve", None), ("实盘 Sleeve", s_live), ("SOTA Sleeve", s_sota)):
            x = bh_returns(w, r, rf, False) + (s if s is not None else 0.0)
            for pn, (a_, z_) in (("2019~2026", ("2019-01-01", "2026-10-01")), ("dev 2019~22", ("2019-01-01", "2022-12-31")), ("test 2023~26", ("2023-01-01", "2026-10-01"))):
                m = (x.index >= a_) & (x.index <= z_); p_ = panel(x[m])
                rows.append(dict(BH配置=f"{key[0]}+{key[1]}", Sleeve=sn, 期间=pn, Sharpe=p_["Sharpe"], CAGR=p_["CAGR"], MDD=p_["MDD"], Calmar=p_["Calmar"], Sortino=p_["Sortino"], 年化波动=p_["年化波动"], CVaR5bp=p_["CVaR5%bp"]))
    D3 = pd.DataFrame(rows); D3.to_csv(TAB / "sota_bh_plus_sleeve.csv", index=False)
    print("\nD3) 2019+:B&H 腿 × {无 / 实盘 / SOTA Sleeve}")
    for pn in ("2019~2026", "dev 2019~22", "test 2023~26"):
        t_ = D3[D3.期间 == pn].pivot_table(index="BH配置", columns="Sleeve", values=["Sharpe", "CAGR", "MDD", "Calmar"], sort=False)
        print(f"\n--- {pn} ---"); print(t_.round(3).to_string())
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
