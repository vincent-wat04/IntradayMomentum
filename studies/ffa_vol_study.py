"""FFA(融合滤波)与 6 个基准在 QQQ 上:统计、分状态、Sleeve 仓位 / 避开低波日 / 波动择时 —— docs/PROPOSAL_ffa_vol.md(预注册)。
python3 ffa_vol_study.py"""
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
from scipy.linalg import solveh_banded
from scipy.optimize import minimize
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowvol_avoidance_study as L
import regime_timing_study as R1
import round3_common as C
import trend_rv_sleeve_study as T
from audit import reimpl_noise as RI
from sens_rv_weights import pct_w, w_linear
import paths as _P

OUT, TAB = L.OUT, L.TAB
W = 1000
LAMS = 10.0 ** np.arange(-1, 4.0001, 0.25)
HS = (1, 5, 22)
OOS0, OOS1 = "2015-01-01", "2026-10-09"
NPERM = 1000


# ============================================================== 数据
def load():
    q = pd.read_parquet(_P.RAW / "yahoo" / "QQQ.parquet"); q.index = pd.to_datetime(q.index).tz_localize(None); q = q.sort_index()
    R = ((np.log(q.high) - np.log(q.low)) ** 2 / (4 * np.log(2)) * 100).astype(float)
    mn = R[R > 0].min(); R = R.where(R > 0, mn)
    r = np.log(q.close).diff().fillna(0.0)
    return q, R, r


# ============================================================== FFA-P(论文原样,逐日窗口双侧平滑)
def ffa_paper(y, start):
    """y = ln R(numpy)。返回 {h: 预测数组}(预测对齐在『目标日』上:F_h[T+h] = 第 T 日形成的预测)及所选 λ。"""
    n = len(y); F = {h: np.full(n, np.nan) for h in HS}; lam_sel = np.full(n, np.nan)
    ab = {}
    for lam in LAMS:
        d = np.full(W, 1 + 2 * lam); d[0] = d[-1] = 1 + lam; u = np.full(W, -lam); u[0] = 0.0
        ab[lam] = np.vstack([u, d])
    for T_ in range(max(start, W - 1), n - 1):
        yw = y[T_ - W + 1:T_ + 1]; best = {h: (np.inf, None, None) for h in HS}
        for lam in LAMS:
            s = solveh_banded(ab[lam], yw, check_finite=False)
            for h in HS:
                e = yw[h:] - s[:-h]; rss = float(e @ e)
                if rss < best[h][0]: best[h] = (rss, lam, s)
        for h in HS:
            _, lam, s = best[h]
            if T_ + h < n:
                Cf = float(np.mean(np.exp(s[h:] - s[:-h])))
                F[h][T_ + h] = np.exp(s[-1]) * Cf
            if h == 1: lam_sel[T_ + 1] = lam
    return F, lam_sel


# ============================================================== FFA-C(严格因果:一侧滤波 = 对数 EWMA)
def gain(lam):
    q = 1.0 / lam; return (-q + np.sqrt(q * q + 4 * q)) / 2


def onesided(y, k):
    zi = np.array([(1 - k) * y[0]])
    return lfilter([k], [1, -(1 - k)], y, zi=zi)[0]


def roll_sum_causal(x, w):
    """x[t] 已对齐在『可得时点』;返回截至 t(含)的过去 w 个的和。"""
    c = np.nancumsum(np.where(np.isfinite(x), x, 0.0)); out = c.copy(); out[w:] = c[w:] - c[:-w]; out[:w - 1] = np.nan
    return out


def ffa_causal(y, R, sd_state=False):
    """返回 {h: F}。每个候选 λ(若 sd_state,还有 k_hi)跑一侧滤波;在过去 W 个『已实现的』一步误差上选。"""
    n = len(y); cands = []
    for lam in LAMS:
        k = gain(lam)
        if not sd_state: cands.append((lam, k, None, onesided(y, k)))
        else:
            base = onesided(y, k); innov = np.r_[np.nan, y[1:] - base[:-1]]
            su = pd.Series(innov).rolling(W, min_periods=250).std().shift(1).to_numpy()
            for khi in (min(2 * k, 0.95), min(4 * k, 0.95), 0.5, 0.8):
                s = np.empty(n); s[0] = y[0]
                for t in range(1, n):
                    e = y[t] - s[t - 1]; kk = khi if (np.isfinite(su[t]) and e > 1.5 * su[t]) else k
                    s[t] = s[t - 1] + kk * e
                cands.append((lam, k, khi, s))
    F = {}
    for h in HS:
        errs = []; js = []
        for (_, _, _, s) in cands:
            e2 = np.full(n, np.nan); e2[h:] = (y[h:] - s[:-h]) ** 2          # 在 t+h 时可得
            errs.append(roll_sum_causal(e2, W))
            jr = np.full(n, np.nan); jr[h:] = R[h:] / np.exp(s[:-h]); js.append(pd.Series(jr).rolling(W, min_periods=250).mean().to_numpy())
        E_ = np.vstack(errs); J_ = np.vstack(js); S_ = np.vstack([c[3] for c in cands])
        Fh = np.full(n, np.nan)
        for T_ in range(W, n - h):
            col = E_[:, T_]
            if not np.isfinite(col).any(): continue
            i = int(np.nanargmin(col)); Fh[T_ + h] = np.exp(S_[i, T_]) * J_[i, T_]
        F[h] = Fh
    return F


# ============================================================== 参数模型(每年初在过去 W 日上 QMLE 估计,年内递推)
def carr_filter(p, R, rneg, lam0):
    w_, a, b_, th = p; n = len(R); lam = np.empty(n); lam[0] = lam0
    for t in range(1, n): lam[t] = w_ + a * R[t - 1] + b_ * lam[t - 1] + th * R[t - 1] * rneg[t - 1]
    return lam


def carr_fit(R, rneg):
    m = R.mean()
    def nll(z):
        a, b_, th = np.exp(z[1:4]) / (1 + np.exp(z[1:4])) * np.array([0.5, 1.0, 0.5])
        if a + b_ + th / 2 >= 0.999: return 1e10
        w_ = m * (1 - a - b_ - th / 2); lam = carr_filter((w_, a, b_, th), R, rneg, m)
        return float(np.sum(np.log(lam) + R / lam))
    best = None
    for z0 in ([0, -1.4, 1.5, -2.0], [0, -0.8, 1.0, -3.0]):
        r_ = minimize(nll, z0, method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r_.fun < best.fun: best = r_
    a, b_, th = np.exp(best.x[1:4]) / (1 + np.exp(best.x[1:4])) * np.array([0.5, 1.0, 0.5]); w_ = m * (1 - a - b_ - th / 2)
    return (w_, a, b_, th)


def ccarr_filter(p, R, lam0):
    om, rho, phi, a, b_ = p; n = len(R); lam = np.empty(n); q = np.empty(n); lam[0] = q[0] = lam0
    for t in range(1, n):
        q[t] = om + rho * q[t - 1] + phi * (R[t - 1] - lam[t - 1])
        lam[t] = q[t] + a * (R[t - 1] - q[t - 1]) + b_ * (lam[t - 1] - q[t - 1])
    return np.maximum(lam, 1e-8), q


def ccarr_fit(R):
    m = R.mean()
    def unpack(z):
        rho = 0.95 + 0.0499 / (1 + np.exp(-z[0])); phi = 0.2 / (1 + np.exp(-z[1])); a = 0.5 / (1 + np.exp(-z[2])); b_ = 0.98 / (1 + np.exp(-z[3]))
        return (m * (1 - rho), rho, phi, a, b_)
    def nll(z):
        p = unpack(z)
        if p[3] + p[4] >= 0.995: return 1e10
        lam, _ = ccarr_filter(p, R, m); return float(np.sum(np.log(lam) + R / lam))
    best = None
    for z0 in ([2, -2, -1, 0.5], [1, -1, 0, 0]):
        r_ = minimize(nll, z0, method="Nelder-Mead", options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r_.fun < best.fun: best = r_
    return unpack(best.x)


def spline_basis(n, step=250):
    t = np.arange(n) / n; knots = np.arange(step, n, step) / n
    return np.column_stack([np.ones(n), t, t ** 2] + [np.maximum(t - k, 0) ** 2 for k in knots])


def rgarch_filter(p, r, x, lh0):
    om, be, ga = p[:3]; n = len(r); lh = np.empty(n); lh[0] = lh0
    for t in range(1, n): lh[t] = om + be * lh[t - 1] + ga * x[t - 1]
    return lh


def rgarch_fit(r, x):
    lh0 = np.log(np.var(r))
    def nll(z):
        om, be, ga, xi, phi, t1, t2, ls = z
        if not (0 < be < 1 and 0 <= ga < 1 and be + ga * phi < 0.9999): return 1e10
        lh = rgarch_filter(z, r, x, lh0); h = np.exp(lh); zz = r / np.sqrt(h)
        u = x - xi - phi * lh - t1 * zz - t2 * (zz ** 2 - 1); s2 = np.exp(2 * ls)
        return float(0.5 * np.sum(lh + zz ** 2) + 0.5 * np.sum(np.log(s2) + u ** 2 / s2))
    z0 = [0.1, 0.55, 0.4, 0.0, 1.0, -0.05, 0.05, np.log(0.4)]
    r_ = minimize(nll, z0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-5})
    return r_.x


def param_forecasts(R, r, y, yrs_start=2004):
    """每年初用过去 W 日估计参数,对该年逐日一步预测(以及 CARR/成分 CARR/HAR/EWMA 的 h=5、22)。"""
    dates = R.index; n = len(R); Rv = R.to_numpy(); rv = r.to_numpy(); rneg = (rv < 0).astype(float)
    out = {m: {h: np.full(n, np.nan) for h in HS} for m in ("EWMA", "HAR", "CARR", "成分CARR")}
    out["Spline-CARR"] = {1: np.full(n, np.nan)}; out["RealGARCH"] = {1: np.full(n, np.nan)}
    # EWMA(全序列递推,无估计)
    ew = np.empty(n); ew[0] = Rv[0]
    for t in range(1, n): ew[t] = 0.94 * ew[t - 1] + 0.06 * Rv[t - 1]           # ew[t] = 对 t 日的预测
    for h in HS: out["EWMA"][h][:] = np.r_[np.full(h - 1, np.nan), ew[:n - h + 1]]
    R5 = pd.Series(Rv).rolling(5).mean().to_numpy(); R22 = pd.Series(Rv).rolling(22).mean().to_numpy()
    xr = np.log(Rv / 100.0); rdm = rv - np.mean(rv)
    for yr in range(yrs_start, 2027):
        i0 = int(np.searchsorted(dates, pd.Timestamp(f"{yr}-01-01"))); i1 = int(np.searchsorted(dates, pd.Timestamp(f"{yr + 1}-01-01")))
        if i0 >= n or i0 - W < 30: continue
        a_, b_ = i0 - W, i0                      # 估计窗口 [a_, b_)
        # HAR(直接法,各 h)
        for h in HS:
            idx = np.arange(max(a_, 22), b_ - h)
            Xh = np.column_stack([np.ones(len(idx)), Rv[idx], R5[idx], R22[idx]]); bb = np.linalg.lstsq(Xh, Rv[idx + h], rcond=None)[0]
            for T_ in range(i0 - h, min(i1, n) - h):
                out["HAR"][h][T_ + h] = max(bb @ np.array([1, Rv[T_], R5[T_], R22[T_]]), Rv[a_:b_].min())
        # CARR
        p = carr_fit(Rv[a_:b_], rneg[a_:b_]); seg = slice(a_, min(i1, n)); lam = carr_filter(p, Rv[seg], rneg[seg], Rv[a_:b_].mean())
        pi = p[1] + p[2] + p[3] / 2; mu = p[0] / (1 - pi); off = a_
        for t in range(i0, min(i1, n)):
            out["CARR"][1][t] = lam[t - off]
            for h in (5, 22):
                if t - 1 + h < n and t - 1 >= off: out["CARR"][h][t - 1 + h] = mu + pi ** (h - 1) * (lam[t - off] - mu)
        # 成分 CARR
        pc = ccarr_fit(Rv[a_:b_]); lamc, qc = ccarr_filter(pc, Rv[seg], Rv[a_:b_].mean()); om, rho, phi, a1, b1 = pc; qbar = om / (1 - rho)
        for t in range(i0, min(i1, n)):
            out["成分CARR"][1][t] = lamc[t - off]
            for h in (5, 22):
                if t - 1 + h < n:
                    qh = qbar + rho ** (h - 1) * (qc[t - off] - qbar); out["成分CARR"][h][t - 1 + h] = qh + (a1 + b1) ** (h - 1) * (lamc[t - off] - qc[t - off])
        # Spline-CARR(两步)
        Bm = spline_basis(W); co = np.linalg.lstsq(Bm, np.log(Rv[a_:b_]), rcond=None)[0]; tau_end = float(np.exp(Bm[-1] @ co)); tau = np.exp(Bm @ co)
        Rn = Rv[a_:b_] / tau; ps = carr_fit(Rn, np.zeros(W)); Rn_all = np.r_[Rn, Rv[b_:min(i1, n)] / tau_end]
        lam_s = carr_filter(ps, Rn_all, np.zeros(len(Rn_all)), Rn.mean())
        for t in range(i0, min(i1, n)): out["Spline-CARR"][1][t] = tau_end * lam_s[t - off]
        # RealGARCH(对数线性)
        z = rgarch_fit(rdm[a_:b_], xr[a_:b_]); lh = rgarch_filter(z, rdm[seg], xr[seg], np.log(np.var(rdm[a_:b_])))
        om_, be_, ga_, xi, phi_, t1, t2, ls = z; s2 = np.exp(2 * ls); bq = t2
        mgf = (1 - 2 * bq) ** -0.5 * np.exp(t1 ** 2 / (2 * (1 - 2 * bq))) if bq < 0.5 else np.nan
        for t in range(i0, min(i1, n)): out["RealGARCH"][1][t] = 100 * np.exp(xi + phi_ * lh[t - off] - t2 + s2 / 2) * mgf
        print(f"   {yr}: CARR α+β+θ/2={pi:.3f};成分CARR ρ={rho:.4f};RealGARCH β={be_:.3f} γ={ga_:.3f} φ={phi_:.2f}", flush=True)
    return out


# ============================================================== 评价
def losses(Rt, F):
    m = np.isfinite(Rt) & np.isfinite(F) & (F > 0)
    e = Rt[m] - F[m]; q = Rt[m] / F[m] - np.log(Rt[m] / F[m]) - 1
    return m, e, q


def dm(d, lag):
    d = d[np.isfinite(d)]; n = len(d); mu = d.mean(); g = [np.mean((d[k:] - mu) * (d[:n - k] - mu)) for k in range(lag + 1)]
    v = g[0] + 2 * sum((1 - k / (lag + 1)) * g[k] for k in range(1, lag + 1)); return mu / np.sqrt(v / n)


def stat_tables(R, FC, dates):
    Rt = R.to_numpy(); oos = (dates >= OOS0) & (dates <= OOS1); rows = []
    q20, q80 = np.nanquantile(Rt[oos], [0.2, 0.8])
    for h in HS:
        common = oos.copy()
        for m_, d_ in FC.items():
            if h in d_: common &= np.isfinite(d_[h]) & (d_[h] > 0)
        ref = FC["FFA-P"][h]
        for m_, d_ in FC.items():
            if h not in d_: continue
            F = d_[h]; row = dict(h=h, 模型=m_, n=int(common.sum()))
            for st, mk in (("全", common), ("低波(≤20%)", common & (Rt <= q20)), ("正常", common & (Rt > q20) & (Rt <= q80)), ("高波(>80%)", common & (Rt > q80))):
                e = Rt[mk] - F[mk]; q = Rt[mk] / F[mk] - np.log(Rt[mk] / F[mk]) - 1
                row[f"RMSE_{st}"] = np.sqrt(np.mean(e ** 2)); row[f"MAE_{st}"] = np.mean(np.abs(e)); row[f"QLIKE_{st}"] = np.mean(q)
            if m_ != "FFA-P":
                for ln, f_ in (("SE", lambda a, b: (a - b) ** 2), ("AE", lambda a, b: np.abs(a - b)), ("QL", lambda a, b: a / b - np.log(a / b) - 1)):
                    d = f_(Rt[common], F[common]) - f_(Rt[common], ref[common]); row[f"DM_{ln}(vs FFA-P)"] = dm(d, h + 5)      # >0:该模型损失更大(FFA 更好)
            hi = common & (Rt > q80); F_ = F[common]
            row["高波日 F/R 中位"] = float(np.median(F[hi] / Rt[hi])); row["高波日被预测为平静(F 在自身最低40%)"] = float(np.mean(F[hi] <= np.quantile(F_, 0.4)))
            lo = common & (Rt <= q20); row["低波日 F/R 中位"] = float(np.median(F[lo] / Rt[lo]))
            rows.append(row)
    return pd.DataFrame(rows)


def panel(x, name=""):
    from scipy.stats import kurtosis, skew
    x = pd.Series(x).dropna(); d = x[x < 0]; eq = (1 + x).cumprod(); mdd = float((eq / eq.cummax() - 1).min()); yrs = len(x) / 252
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else np.nan; dn = np.sqrt((np.minimum(x, 0) ** 2).mean()); prof = x[x > 0].sum()
    return {"方案": name, "均值bp": 1e4 * x.mean(), "中位bp": 1e4 * x.median(), "Sharpe": x.mean() / x.std() * np.sqrt(252), "Sortino": x.mean() / dn * np.sqrt(252) if dn > 0 else np.nan,
            "CAGR": cagr, "MDD": mdd, "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "偏度": skew(x), "峰度": kurtosis(x), "CVaR5%bp": 1e4 * x[x <= x.quantile(0.05)].mean(),
            "盈亏比": prof / abs(d.sum()) if len(d) else np.nan, "前5%日占利润": x.nlargest(max(int(.05 * len(x)), 1)).sum() / prof if prof > 0 else np.nan, "最差日bp": 1e4 * x.min()}


def main():
    t0 = time.time(); pd.set_option("display.width", 260, "display.max_rows", 300, "display.float_format", lambda x: f"{x:9.4f}")
    q, R, r = load(); dates = R.index; y = np.log(R.to_numpy()); Rv = R.to_numpy(); n = len(R)
    start = int(np.searchsorted(dates, pd.Timestamp("2004-01-01")))
    print(f"QQQ 日 OHLC {dates[0].date()}~{dates[-1].date()} n={n};R(Parkinson×100)中位 {np.median(Rv):.4f}")
    # ---- 自检:第 T 日之后的 R 全部 ×7.3,T+1 的预测不得改变(FFA-P、FFA-C);阳性对照:把 R_{T+1} 放进窗口必须改变
    T_chk = int(np.searchsorted(dates, pd.Timestamp("2016-06-30"))); y2 = y.copy(); y2[T_chk + 1:] += np.log(7.3)
    def one_P(yy, T_):
        best = (np.inf, None)
        for lam in LAMS:
            d = np.full(W, 1 + 2 * lam); d[0] = d[-1] = 1 + lam; u = np.full(W, -lam); u[0] = 0.0; s = solveh_banded(np.vstack([u, d]), yy[T_ - W + 1:T_ + 1])
            e = yy[T_ - W + 2:T_ + 1] - s[:-1]; rs = float(e @ e)
            if rs < best[0]: best = (rs, np.exp(s[-1]) * np.mean(np.exp(s[1:] - s[:-1])))
        return best[1]
    a0, a1, lk = one_P(y, T_chk), one_P(y2, T_chk), one_P(y2, T_chk + 1)
    Fc0 = ffa_causal(y, Rv)[1][T_chk + 1]; Fc1 = ffa_causal(y2, np.exp(y2))[1][T_chk + 1]
    print(f"[自检] T=2016-06-30 之后 R×7.3:FFA-P 对 T+1 的预测 {a0:.6f} vs {a1:.6f}(须相同);FFA-C {Fc0:.6f} vs {Fc1:.6f}(须相同);阳性对照(窗口含 T+1)FFA-P {lk:.6f}(须不同)")
    assert abs(a0 - a1) < 1e-12 and abs(Fc0 - Fc1) < 1e-12 and abs(lk - a0) > 1e-6
    # ---- 预测
    print("FFA-P(论文版,逐日窗口)…", flush=True); FP, lam_sel = ffa_paper(y, start); print(f"   完成 {time.time()-t0:.0f}s;所选 λ 中位 {np.nanmedian(lam_sel):.1f}(对应一侧增益 k={gain(np.nanmedian(lam_sel)):.3f})")
    FCz = ffa_causal(y, Rv); print(f"FFA-C 完成 {time.time()-t0:.0f}s", flush=True)
    FSD = ffa_causal(y, Rv, sd_state=True); print(f"FFA-SD 完成 {time.time()-t0:.0f}s", flush=True)
    print("参数模型(每年重估)…", flush=True); PM = param_forecasts(R, r, y); print(f"   完成 {time.time()-t0:.0f}s")
    FC = {"FFA-P": FP, "FFA-C": FCz, "FFA-SD": FSD, "FFA-MAX": {h: np.fmax(FCz[h], PM["EWMA"][h]) for h in HS}, **PM}
    FC["FFA-MAX"] = {h: np.where(np.isfinite(FCz[h]) & np.isfinite(PM["EWMA"][h]), np.maximum(FCz[h], PM["EWMA"][h]), np.nan) for h in HS}
    pd.to_pickle(dict(FC=FC, dates=dates, R=R, lam=lam_sel), OUT / "ffa_forecasts.pkl")
    # ==================================================== 统计
    S = stat_tables(R, FC, dates); S.to_csv(TAB / "ffa_stat.csv", index=False)
    for h in HS:
        s = S[S.h == h]; print(f"\n=== 统计(OOS {OOS0[:4]}–2026,h={h},n={int(s.n.iloc[0])};R 的单位 = Parkinson 方差×100)===")
        print(s[["模型", "RMSE_全", "MAE_全", "QLIKE_全", "MAE_低波(≤20%)", "MAE_正常", "MAE_高波(>80%)", "RMSE_高波(>80%)", "QLIKE_高波(>80%)"] + [c for c in s.columns if c.startswith("DM_")] + ["高波日 F/R 中位", "高波日被预测为平静(F 在自身最低40%)", "低波日 F/R 中位"]].to_string(index=False))
    S21 = None
    # 2021+ 子样本(与 Sleeve 重叠)
    global OOS0_
    # ==================================================== 经济:Sleeve(2019–2026)
    from etft.intraday.bars import load_bars
    b = load_bars("QQQ"); days = b.days
    base = pd.read_pickle(OUT / "sleeve_anatomy_curve.pkl"); base.index = pd.to_datetime(base.index); base = base.reindex(days).fillna(0.0)
    sd14 = RI.daily_sigma(b.day_close, b.prev_close_adj, 14).reindex(days); expo0 = np.minimum(4.0, 0.02 / sd14)
    rcc = pd.Series(np.log(b.day_close / b.prev_close_adj), index=days)
    td = pd.Series(days.dayofweek != 0, index=days); yrs = days.year
    import disp_pred_study as DP
    a_ = DP.Arr(b); dn = DP.targets(a_, 30).dn; hi_dn = dn >= dn[days.year >= 2019].quantile(0.8)
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    port = lambda sl: T.build(bh, sl.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px_all, p_all)[0]
    base_p = port(base); sel = td.to_numpy() & (yrs >= 2019)
    rows, pan, ratios = [], [panel(base[td], "基线 Sleeve(σ₁₄)"), panel(base_p, "基线 组合")], {}
    for m_ in ["FFA-P", "FFA-C", "FFA-SD", "FFA-MAX", "EWMA", "HAR", "CARR", "成分CARR", "Spline-CARR", "RealGARCH"]:
        F1 = pd.Series(FC[m_][1], index=dates).reindex(days)
        kap = np.sqrt((rcc ** 2).rolling(252, min_periods=126).mean().shift(1) / (F1 / 100).rolling(252, min_periods=126).mean().shift(1))
        sig = np.sqrt(F1 / 100) * kap; ex = np.minimum(4.0, 0.02 / sig); ratio = (ex / expo0).where(np.isfinite(ex) & np.isfinite(expo0), 1.0).fillna(1.0)
        ratios[m_] = ratio; xs = base * ratio; xp = port(xs)
        row = dict(模型=m_, 平均expo比=float(ratio[sel].mean()), expo触顶4x占比=float((ex[sel] >= 3.999).mean()), 基线触顶占比=float((expo0[sel] >= 3.999).mean()),
                   高位移日expo比=float(ratio[sel & hi_dn.to_numpy()].mean()), 低位移日expo比=float(ratio[sel & (dn <= dn[yrs >= 2019].quantile(0.2)).to_numpy()].mean()),
                   高位移日Sleeve均bp_基线=1e4 * float(base[sel & hi_dn.to_numpy()].mean()), 高位移日Sleeve均bp_新=1e4 * float(xs[sel & hi_dn.to_numpy()].mean()))
        for k in ("dev", "test", "全"):
            mk = sel & ((yrs <= 2022) if k == "dev" else (yrs >= 2023) if k == "test" else (yrs >= 2019))
            row[f"Sleeve ΔSR {k}"] = L.sr(xs[mk]) - L.sr(base[mk])
            pm = (xp.index.year <= 2022) if k == "dev" else (xp.index.year >= 2023) if k == "test" else (xp.index.year >= 2019)
            row[f"组合 ΔSR {k}"] = L.sr(xp[pm & (xp.index >= "2019-01-01")]) - L.sr(base_p[pm & (base_p.index >= "2019-01-01")])
        rows.append(row); pan.append(panel(xs[td], f"{m_} [Sleeve]")); pan.append(panel(xp, f"{m_} [组合]"))
    # 置换(圆周平移 expo 比,≥60 日),族内 max(10)
    rg = np.random.default_rng(0); idx = np.where(sel)[0]; bsr = L.sr(base.iloc[idx]); shifts = rg.integers(60, len(idx) - 60, NPERM); nulls = {}
    for m_, ratio in ratios.items():
        rv_ = ratio.to_numpy()[idx]; bv = base.to_numpy()[idx]; nulls[m_] = np.array([L.sr(pd.Series(bv * np.roll(rv_, s))) for s in shifts]) - bsr
    fam = np.max(np.vstack(list(nulls.values())), axis=0)
    E1 = pd.DataFrame(rows); E1["置换p(族内max,10)"] = [float(np.mean(fam >= (L.sr(base.iloc[idx] * ratios[m_].iloc[idx]) - bsr))) for m_ in E1.模型]
    print("\n=== E1 Sleeve 仓位:用各模型的预测替换 σ₁₄(仅改 expo,交易逻辑不变;2019–2026,周一排除)==="); print(E1.to_string(index=False)); E1.to_csv(TAB / "ffa_e1_sizing.csv", index=False)
    # ==================================================== E2 避开低波日(描述性)
    rows = []
    oracle = pd.Series(Rv, index=dates).reindex(days)
    for m_ in ["FFA-P", "FFA-C", "FFA-SD", "HAR", "EWMA", "成分CARR", "神谕(真实R)"]:
        F1 = oracle if m_.startswith("神谕") else pd.Series(FC[m_][1], index=dates).reindex(days)
        for kind in ("绝对", "相对σ₁₄²"):
            sig = F1 if kind == "绝对" else F1 / (100 * sd14 ** 2)
            rk = L.causal_s(sig.where(days.year >= 2018), "skip", 0.2); xs = base * rk
            rows.append(dict(模型=m_, 口径=kind, 跳过比例=float((rk[sel] == 0).mean()), **{f"Sleeve ΔSR {k}": L.sr(xs[sel & mk]) - L.sr(base[sel & mk]) for k, mk in (("dev", yrs <= 2022), ("test", yrs >= 2023), ("全", yrs >= 2019))},
                             被跳过日Sleeve均bp=1e4 * float(base[sel & (rk == 0).to_numpy()].mean()), 被跳过日中高位移占比=float(hi_dn[sel & (rk == 0).to_numpy()].mean())))
    E2 = pd.DataFrame(rows); print("\n=== E2 避开预测的低波日(因果排名 ≤20% → 当日 Sleeve 不交易)==="); print(E2.to_string(index=False)); E2.to_csv(TAB / "ffa_e2_skip.csv", index=False)
    # ==================================================== E3 论文的波动择时(QQQ B&H 层,2004-06+)
    rq = px_all.pct_change().fillna(0.0); rows = []
    w_rv = w_linear(p_all, 0.5, 1.5).shift(1).fillna(1.0)
    for m_ in ["静态", "RV21[0.5,1.5]", "FFA-P", "FFA-C", "FFA-SD", "EWMA", "HAR", "CARR", "成分CARR", "Spline-CARR", "RealGARCH"]:
        if m_ == "静态": w = pd.Series(1.0, index=rq.index)
        elif m_.startswith("RV21"): w = w_rv
        else:
            F1 = pd.Series(FC[m_][1], index=dates).reindex(rq.index); sv = np.sqrt(F1); c_ = sv.rolling(252, min_periods=126).median().shift(1)
            w = np.minimum(1.5, c_ / sv).fillna(1.0)                    # F1[t] 是 t−1 收盘形成的对 t 日的预测 → 已是 t 日可用的权重
        for wn, (a, z) in {"2004-06+": ("2004-06-01", "2026-10-09"), "2019+": ("2019-01-01", "2026-10-09")}.items():
            mk = (rq.index >= a) & (rq.index <= z); x = R1.strat(w, rq)[mk]; ctrl = float(w[mk].mean()) * rq[mk]
            d, dist = T.boot_dsr(x, ctrl, nb=1000, L=40, seed=0)
            rows.append(dict(权重=m_, 期间=wn, 平均权重=float(w[mk].mean()), 日均换手=float(w[mk].diff().abs().mean()), Sharpe=R1.sr(x), CAGR=R1.cagr(x), MDD=R1.mdd(x), dSR_vs同暴露静态=d, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5)))
    E3 = pd.DataFrame(rows); print("\n=== E3 论文的波动择时 w=min(1.5, c/√F)(QQQ,成本 2bp/边)==="); print(E3.to_string(index=False)); E3.to_csv(TAB / "ffa_e3_timing.csv", index=False)
    Pn = pd.DataFrame(pan); print("\n=== 分布面板(Sleeve 与组合,2019–2026)==="); print(Pn.to_string(index=False)); Pn.to_csv(TAB / "ffa_panel.csv", index=False)
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
