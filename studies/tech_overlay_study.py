"""日内『做T』叠加仓(倒T/正T,事件驱动)—— docs/PROPOSAL_intraday_t_overlay.md(预注册)。
python3 tech_overlay_study.py [selftest|run]"""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intraday_tech_study as T
from tech_indicators import wilder, sma

OUT, TAB = T.OUT, T.TAB
COSTS, HEAD, SPLIT, NPERM, SEED = T.COSTS, T.HEAD, T.SPLIT, T.NPERM, T.SEED
MAXENT = 3
JMIN = {5: 5, 15: 1}                 # 首次入场:10:00 收盘
JMAX = {5: 71, 15: 23}               # 最晚入场:15:30 收盘
# 指标: (高区, 中枢, 低区)
THR = {"BB": (2.0, 0.0, -2.0), "RSI": (70, 50, 30), "KDJ": (80, 50, 20), "CCI": (100, 0, -100), "WR": (-20, -50, -80),
       "MFI": (80, 50, 20), "POS": (0.9, 0.5, 0.1), "VWAP": (1.0, 0.0, -1.0)}
NAMES = list(THR)
SIDES = ("倒T", "正T"); EXITS = ("E1", "E2")


def indicators(ob, hb, lb, cb, vb, leak_rsi=0):
    """(D,B) 的 OHLCV -> {name: 展平序列}。全部因果;leak_rsi>0 时 RSI 偷看未来 leak 根(阳性对照)。"""
    D, B = ob.shape
    C, H, L, V = (pd.Series(x.ravel()) for x in (cb, hb, lb, vb))
    out = {}
    m = sma(C, 20); sd = C.rolling(20).std(ddof=0); out["BB"] = ((C - m) / sd.replace(0, np.nan)).to_numpy()
    Cr = C.shift(-leak_rsi) if leak_rsi else C
    d = Cr.diff(); g = wilder(d.clip(lower=0).fillna(0), 14); ls = wilder((-d).clip(lower=0).fillna(0), 14); out["RSI"] = (100 * g / (g + ls)).to_numpy()
    ll9, hh9 = L.rolling(9).min(), H.rolling(9).max(); rsv = 100 * (C - ll9) / (hh9 - ll9).replace(0, np.nan)
    out["KDJ"] = rsv.ewm(alpha=1 / 3, adjust=False).mean().to_numpy()
    tp = (H + L + C) / 3; mt = tp.rolling(14).mean(); mad = tp.rolling(14).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
    out["CCI"] = ((tp - mt) / (0.015 * mad.replace(0, np.nan))).to_numpy()
    hh, ll = H.rolling(14).max(), L.rolling(14).min(); out["WR"] = (-100 * (hh - C) / (hh - ll).replace(0, np.nan)).to_numpy()
    dtp = tp.diff(); mf = tp * V; pos_ = mf.where(dtp > 0, 0.0).rolling(14).sum(); neg_ = mf.where(dtp < 0, 0.0).rolling(14).sum()
    out["MFI"] = (100 * pos_ / (pos_ + neg_).replace(0, np.nan)).to_numpy()
    hh20, ll20 = H.rolling(20).max(), L.rolling(20).min(); out["POS"] = ((C - ll20) / (hh20 - ll20).replace(0, np.nan)).to_numpy()
    # VWAP 偏离(当日累计;σ14 = 前 14 个交易日日收益标准差,shift 1)
    tpd = (hb + lb + cb) / 3.0; cv = np.cumsum(vb, axis=1); vw = np.cumsum(tpd * vb, axis=1) / np.where(cv > 0, cv, np.nan)
    vw = np.where(np.isfinite(vw), vw, tpd)
    dc = pd.Series(cb[:, -1]); sd14 = np.log(dc).diff().rolling(14).std().shift(1).to_numpy()
    out["VWAP"] = (np.log(cb / vw) / (0.25 * sd14[:, None])).ravel()
    return out


def run_machine(entry, exit_, B, jmin, jmax, sign, maxent=MAXENT):
    n = len(entry); P = np.zeros(n, np.int8); st = 0; nent = 0; tr = []; cur = None
    en, ex = entry.tolist(), exit_.tolist()
    for t in range(n):
        j = t % B
        if j == 0:
            st = 0; nent = 0
        if j > B - 2:
            continue
        if st != 0 and ex[t]:
            tr.append((t // B, cur, j, False)); st = 0
        if st == 0 and en[t] and jmin <= j <= jmax and nent < maxent:
            st = sign; nent += 1; cur = j
        P[t] = st
        if j == B - 2 and st != 0:
            tr.append((t // B, cur, B - 1, True))            # 持有到最后一根,收盘强制了结
    return P, tr


def make_configs(ind, B, k):
    cfgs = {}
    for nm in NAMES:
        x = pd.Series(ind[nm]); hi, mid, lo = THR[nm]; xp = x.shift(1)
        hi_cross = ((x > hi) & (xp <= hi)).to_numpy(); lo_cross = ((x < lo) & (xp >= lo)).to_numpy()
        for side in SIDES:
            for ex in EXITS:
                if side == "倒T":
                    en = hi_cross; xt = (x < mid).to_numpy() if ex == "E1" else (x < lo).to_numpy(); sg = -1
                else:
                    en = lo_cross; xt = (x > mid).to_numpy() if ex == "E1" else (x > hi).to_numpy(); sg = 1
                cfgs[(k, nm, side, ex)] = (en, xt, sg)
    return cfgs


def positions_for(ind, D, B, k):
    out = {}
    for key, (en, xt, sg) in make_configs(ind, B, k).items():
        P, tr = run_machine(en, xt, B, JMIN[k], JMAX[k], sg)
        out[key] = (P.reshape(D, B), tr)
    return out


def trade_stats(tr, ob, cb, sg):
    if not tr: return dict(n_trades=0)
    d = np.array([t[0] for t in tr]); ji = np.array([t[1] for t in tr]); jo = np.array([t[2] for t in tr]); forced = np.array([t[3] for t in tr])
    pin = ob[d, ji + 1]; pout = np.where(forced, cb[d, -1], ob[d, np.minimum(jo + 1, ob.shape[1] - 1)])
    g = sg * (pout / pin - 1) * 1e4
    return dict(n_trades=len(tr), trade_gross_bp=g.mean(), hit=(g > 0).mean(), forced_share=forced.mean(),
                forced_gross_bp=g[forced].mean() if forced.any() else np.nan, signal_gross_bp=g[~forced].mean() if (~forced).any() else np.nan,
                worst_trade_bp=g.min())


def selftest(days, o, h, l, c, v):
    print("== 自检:未来污染 ==")
    ok_all = True
    for k in (5, 15):
        ob, hb, lb, cb, vb = T.bars_k(o, h, l, c, v, k); D, B = ob.shape
        K = (D // 2) * B + B // 2
        i0 = indicators(ob, hb, lb, cb, vb); P0 = positions_for(i0, D, B, k)
        rng = np.random.default_rng(1); bad = []
        for x in (ob, hb, lb, cb, vb):
            y = x.ravel().copy(); y[K + 1:] = 100 + 50 * rng.random(len(y) - K - 1); bad.append(y.reshape(D, B))
        hi_, lo_ = np.maximum(bad[1], bad[2]), np.minimum(bad[1], bad[2]); bad[1], bad[2] = hi_, lo_
        i1 = indicators(*bad); P1 = positions_for(i1, D, B, k)
        dind = {nm: int((np.nan_to_num(i0[nm][:K + 1], nan=-9e9) != np.nan_to_num(i1[nm][:K + 1], nan=-9e9)).sum()) for nm in NAMES}
        dpos = {key: int((P0[key][0].ravel()[:K + 1] != P1[key][0].ravel()[:K + 1]).sum()) for key in P0}
        ok = sum(dind.values()) == 0 and sum(dpos.values()) == 0; ok_all &= ok
        aft = sum(int((P0[key][0].ravel()[K + 1:] != P1[key][0].ravel()[K + 1:]).any()) for key in P0)
        print(f"  {k}min K={K}: ≤K 指标差异 {sum(dind.values())}、持仓差异 {sum(dpos.values())}(均应=0);K 之后持仓确有变化的配置 {aft}/{len(P0)} → {'PASS' if ok else 'FAIL'}")
        il = indicators(ob, hb, lb, cb, vb, leak_rsi=20); il2 = indicators(*bad, leak_rsi=20)
        det = int((np.nan_to_num(il["RSI"][:K + 1], nan=-9e9) != np.nan_to_num(il2["RSI"][:K + 1], nan=-9e9)).sum())
        print(f"  {k}min 阳性对照(偷看未来 20 根的 RSI)≤K 被污染改变 {det} 处(应>0) → {'PASS' if det > 0 else 'FAIL'}"); ok_all &= det > 0
    print("== 自检结论:", "PASS" if ok_all else "FAIL", "==")
    return ok_all


def run():
    t0 = time.time()
    days, o, h, l, c, v = T.load_full(); years = days.year.to_numpy(); D = len(days)
    print(f"完整交易日 {D}  {days[0].date()}~{days[-1].date()}")
    if not selftest(days, o, h, l, c, v): print("自检失败,终止"); return
    dc = c[:, -1]; r_bh = pd.Series(dc).pct_change().fillna(0.0).to_numpy()
    Ps, Rs, rows, obs_sr = {}, {}, [], {}
    for k in (5, 15):
        ob, hb, lb, cb, vb = T.bars_k(o, h, l, c, v, k); B = ob.shape[1]
        Rs[k] = T.bar_returns(ob, cb).astype(np.float32)
        pos = positions_for(indicators(ob, hb, lb, cb, vb), D, B, k)
        for key, (P, tr) in pos.items():
            Pk = P[:, :B - 1].astype(np.float32); Ps[key] = Pk
            r = T.eval_config(Pk, Rs[k], years); sg = -1 if key[2] == "倒T" else 1
            r.update(trade_stats(tr, ob, cb, sg)); r.update(k=key[0], ind=key[1], side=key[2], exit=key[3])
            g = T.gross_pnl(Pk, Rs[k]); t = T.turnover(Pk); n = g - HEAD * t
            r["trades_day"] = r.get("n_trades", 0) / D; r["worst_day_bp"] = float(n.min())
            comb = r_bh + 0.5 * n * 1e-4
            r["dSR_BH_w0.5"] = T.sr(comb) - T.sr(r_bh); r["uplift_bp_year"] = 0.5 * n.mean() * 252
            rows.append(r)
    print(f"持仓/交易构造完成 {time.time()-t0:.0f}s,配置 {len(Ps)}")
    df = pd.DataFrame(rows); keys = list(Ps.keys())
    obs = np.array([df[(df.k == kk[0]) & (df.ind == kk[1]) & (df.side == kk[2]) & (df.exit == kk[3])]["sr@1.0"].iloc[0] for kk in keys])
    rng = np.random.default_rng(SEED); shifts = rng.integers(5, D - 5, NPERM); plc = np.zeros((len(keys), NPERM), np.float32)
    for k in (5, 15):
        idx = [i for i, kk in enumerate(keys) if kk[0] == k]
        Pst = np.stack([Ps[keys[i]] for i in idx]); Tst = np.stack([T.turnover(Ps[keys[i]]) for i in idx]) * HEAD
        for si, s in enumerate(shifts):
            Rr = np.roll(Rs[k][:, 1:], -int(s), axis=0); g = np.einsum("cdj,dj->cd", Pst, Rr) - Tst
            sd_ = g.std(1, ddof=1); plc[idx, si] = np.where(sd_ > 0, g.mean(1) / sd_ * np.sqrt(252), np.nan)
    mx = np.nanmax(plc, axis=0); fin = np.isfinite(obs)
    df["sr_obs"] = [df.loc[(df.k == kk[0]) & (df.ind == kk[1]) & (df.side == kk[2]) & (df.exit == kk[3]), "sr@1.0"].iloc[0] for kk in zip(df.k, df.ind, df.side, df.exit)]
    key_idx = {kk: i for i, kk in enumerate(keys)}
    ii = [key_idx[(a, b, c_, d)] for a, b, c_, d in zip(df.k, df.ind, df.side, df.exit)]
    df["p_single"] = [(1 + np.sum(plc[i] >= obs[i])) / (1 + NPERM) if fin[i] else np.nan for i in ii]
    df["p_family"] = [(1 + np.sum(mx >= obs[i])) / (1 + NPERM) if fin[i] else np.nan for i in ii]
    df["pass_i"] = (df.sr_dev > 0) & (df.sr_test > 0); df["pass_ii"] = df.breakeven >= 2.0; df["pass_iii"] = df.p_family < 0.05
    df["PASS"] = df.pass_i & df.pass_ii & df.pass_iii
    df.to_csv(TAB / "intraday_t_overlay.csv", index=False); pd.to_pickle(dict(df=df, plc_max=mx), OUT / "intraday_t_overlay.pkl")
    report(df, mx); print(f"总耗时 {time.time()-t0:.0f}s")


def report(df, mx):
    pd.set_option("display.width", 260, "display.max_rows", 400, "display.float_format", lambda x: f"{x:8.2f}")
    print(f"\n===== 汇总 =====\n配置 {len(df)};PASS = {int(df.PASS.sum())};族内 max 安慰剂 Sharpe:50%={np.nanpercentile(mx,50):.2f} 95%={np.nanpercentile(mx,95):.2f}")
    print(f"净 SR@1bp>0:{int((df['sr@1.0']>0).sum())};dev&test 同正:{int(df.pass_i.sum())};breakeven≥2bp:{int(df.pass_ii.sum())};p_family<0.05:{int(df.pass_iii.sum())}")
    print("\n-- 按 周期×方向×出场 的中位数 --")
    print(df.groupby(["k", "side", "exit"])[["trades_day", "trade_gross_bp", "hit", "forced_share", "forced_gross_bp", "signal_gross_bp", "gross_bp", "net_bp@1.0", "sr@1.0", "breakeven"]].median())
    cols = ["k", "ind", "side", "exit", "trades_day", "trade_gross_bp", "forced_share", "forced_gross_bp", "signal_gross_bp", "gross_bp", "net_bp@1.0", "sr_gross", "sr@0.25", "sr@1.0", "breakeven", "sr_dev", "sr_test", "p_single", "p_family", "dSR_BH_w0.5", "uplift_bp_year"]
    print("\n-- 净 Sharpe@1bp 最高 12 个 --"); print(df.sort_values("sr@1.0", ascending=False).head(12)[cols].to_string(index=False))
    print("\n-- 毛 Sharpe 最高 8 个 --"); print(df.sort_values("sr_gross", ascending=False).head(8)[cols].to_string(index=False))
    print("\n-- 倒T(用户主问题)全部 32 个,按净 SR 排序 --"); print(df[df.side == "倒T"].sort_values("sr@1.0", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "run"
    if arg == "selftest": d = T.load_full(); selftest(*d)
    else: run()
