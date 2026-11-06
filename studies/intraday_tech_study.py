"""32 个经典技术指标在 QQQ 5/15 分钟 K 线上的日内检验 —— docs/PROPOSAL_intraday_tech_indicators.md(预注册)。
python3 intraday_tech_study.py [selftest|run]"""
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
import tech_indicators as TI
from etft.intraday.bars import load_bars

OUT = Path("outputs/full/intraday"); OUT.mkdir(parents=True, exist_ok=True)
TAB = Path("../outputs/tables") if Path("../outputs/tables").exists() else Path("outputs/tables")
COSTS = (0.25, 0.5, 1.0, 2.0); HEAD = 1.0
MODES = ("LO", "LO_inv", "LS", "LS_inv")
SPLIT = 2023  # dev = 2019–22, test = 2023–26
NPERM = int(os.environ.get("NPERM", 1000)); SEED = 20261011


def load_full():
    b = load_bars("QQQ"); full = b.last_idx.to_numpy() >= 389
    g = lambda x: x.to_numpy()[full]
    return b.o.index[full], g(b.o), g(b.h), g(b.l), g(b.c), g(b.v)


def resample(a, k, how):
    D = a.shape[0]; B = 390 // k; x = a.reshape(D, B, k)
    return {"first": x[:, :, 0], "last": x[:, :, -1], "max": x.max(2), "min": x.min(2), "sum": x.sum(2)}[how]


def bars_k(o, h, l, c, v, k):
    return resample(o, k, "first"), resample(h, k, "max"), resample(l, k, "min"), resample(c, k, "last"), resample(v, k, "sum")


def flat_inputs(ob, hb, lb, cb, vb):
    D, B = ob.shape; nd = np.zeros((D, B), bool); nd[:, 0] = True
    return ob.ravel(), hb.ravel(), lb.ravel(), cb.ravel(), vb.ravel(), nd.ravel()


def bar_returns(ob, cb):
    """Rb[d,b] = bar b 的『开→下一根开』收益(最后一根=开→收盘价),bp。"""
    nxt = np.concatenate([ob[:, 1:], cb[:, -1:]], axis=1)
    return (nxt / ob - 1.0) * 1e4


def positions(sig, D, B, k, mode):
    """sig:(D*B,) 信号(bar 收盘时刻可得)。返回 P[d,j], j=0..B-2:j 收盘决策、持有 bar j+1。首次决策 10:00 收盘。"""
    s = sig.reshape(D, B).astype(np.int8)
    jmin = int(round(30 / k)) * 1 + (0 if k == 5 else 0)
    # 10:00 收盘 = 第 (30/k) 根 bar 的下标(0 起)为 30/k-1;5 分钟 → 5(10:00 为第 6 根下标 5);15 分钟 → 1(10:00 为第 2 根下标 1)
    jmin = {5: 5, 15: 1}[k]
    p = {"LO": (s == 1), "LO_inv": (s == -1), "LS": s, "LS_inv": -s}[mode].astype(np.float32)
    P = p[:, : B - 1].copy(); P[:, :jmin] = 0.0
    return P


def turnover(P):
    Z = np.zeros((P.shape[0], 1), np.float32)
    X = np.concatenate([Z, P, Z], axis=1)
    return np.abs(np.diff(X, axis=1)).sum(1)


def gross_pnl(P, Rb):
    return (P * Rb[:, 1:]).sum(1)


def sr(x):
    x = np.asarray(x, float); s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(252)) if s > 0 else np.nan


def eval_config(P, Rb, years):
    g = gross_pnl(P, Rb); t = turnover(P); res = {"turn_day": t.mean(), "rt_day": t.mean() / 2, "gross_bp": g.mean()}
    for c in COSTS:
        res[f"net_bp@{c}"] = (g - c * t).mean(); res[f"sr@{c}"] = sr(g - c * t)
    res["sr_gross"] = sr(g)
    res["breakeven"] = g.mean() / t.mean() if t.mean() > 0 else np.nan
    n = g - HEAD * t
    for nm, m in (("dev", years < SPLIT), ("test", years >= SPLIT)):
        res[f"sr_{nm}"] = sr(n[m]); res[f"gross_{nm}"] = g[m].mean(); res[f"net_{nm}"] = n[m].mean()
        res[f"be_{nm}"] = g[m].mean() / t[m].mean() if t[m].mean() > 0 else np.nan
    return res


def build_all(days, o, h, l, c, v):
    """返回 {(k,ind,mode): P}, {k: Rb}"""
    Ps, Rs = {}, {}
    for k in (5, 15):
        ob, hb, lb, cb, vb = bars_k(o, h, l, c, v, k); D, B = ob.shape
        Rs[k] = bar_returns(ob, cb).astype(np.float32)
        sig = TI.compute_signals(*flat_inputs(ob, hb, lb, cb, vb), bars_per_day=B)
        for ind in TI.NAMES:
            for m in MODES: Ps[(k, ind, m)] = positions(sig[ind], D, B, k, m)
    return Ps, Rs


# ---------------------------------------------------------------- 自检
def selftest(days, o, h, l, c, v):
    """污染 (d0,j0) 之后所有 bar:≤ 该 bar 的信号必须逐位不变;阳性对照(泄漏 MACD)必须变化。"""
    print("== 自检:未来污染 ==")
    ok_all = True
    for k in (5, 15):
        ob, hb, lb, cb, vb = bars_k(o, h, l, c, v, k); D, B = ob.shape
        base = flat_inputs(ob, hb, lb, cb, vb); nd = base[5]
        K = (D // 2) * B + B // 2   # 中间某天的中间某根
        rng = np.random.default_rng(1)
        bad = [x.copy() for x in base[:5]]
        for x in bad: x[K + 1:] = 100 + 50 * rng.random(len(x) - K - 1)
        hi = np.maximum(bad[1], bad[2]); lo = np.minimum(bad[1], bad[2]); bad[1], bad[2] = hi, lo
        s0 = TI.compute_signals(*base[:5], nd, B); s1 = TI.compute_signals(*bad, nd, B)
        diffs = {ind: int((s0[ind][:K + 1] != s1[ind][:K + 1]).sum()) for ind in TI.NAMES}
        chg_after = sum(int((s0[ind][K + 1:] != s1[ind][K + 1:]).any()) for ind in TI.NAMES)
        ok = all(v_ == 0 for v_ in diffs.values()); ok_all &= ok
        print(f"  {k}min K={K}: ≤K 信号差异总数 {sum(diffs.values())}(应=0)  | K 之后确有变化的指标数 {chg_after}/32(说明污染有效)  → {'PASS' if ok else 'FAIL ' + str({a: b for a, b in diffs.items() if b})}")
        # 阳性对照:偷看未来 20 根收盘的 MACD(单根泄漏只能影响 t=K 一个点,功效不足,故取 20)
        def leaky(cc):
            nxt = np.roll(cc, -20); dif, dea = TI.macd(nxt); return TI._sgn(dif - dea)
        l0, l1 = leaky(base[3]), leaky(bad[3])
        det = int((l0[:K + 1] != l1[:K + 1]).sum())
        print(f"  {k}min 阳性对照(泄漏 MACD)≤K 信号被污染改变 {det} 处(应>0) → {'PASS' if det > 0 else 'FAIL(自检无力)'}")
        ok_all &= det > 0
    print("== 自检结论:", "PASS" if ok_all else "FAIL", "==")
    return ok_all


# ---------------------------------------------------------------- 主流程
def run():
    t0 = time.time()
    days, o, h, l, c, v = load_full(); years = days.year.to_numpy(); D = len(days)
    print(f"完整交易日 {D}  {days[0].date()}~{days[-1].date()}  dev<{SPLIT}≤test")
    if not selftest(days, o, h, l, c, v): print("自检失败,终止"); return
    Ps, Rs = build_all(days, o, h, l, c, v); print(f"信号/持仓构造完成 {time.time()-t0:.0f}s,配置 {len(Ps)}")
    rows = []
    for (k, ind, m), P in Ps.items():
        r = eval_config(P, Rs[k], years); r.update(k=k, ind=ind, mode=m, cls=TI.CLASS[ind]); rows.append(r)
    df = pd.DataFrame(rows)
    # ---- 安慰剂:整天循环错位(持仓 vs 收益),族内 max(净 Sharpe@1bp)
    rng = np.random.default_rng(SEED); shifts = rng.integers(5, D - 5, NPERM)
    keys = list(Ps.keys()); obs = np.array([df.loc[(df.k == k) & (df.ind == i) & (df["mode"] == m), "sr@1.0"].iloc[0] for k, i, m in keys])
    turn = {key: turnover(P) for key, P in Ps.items()}
    plc = np.zeros((len(keys), NPERM), np.float32)
    for k in (5, 15):
        idx = [i for i, key in enumerate(keys) if key[0] == k]
        Pst = np.stack([Ps[keys[i]] for i in idx]); Tst = np.stack([turn[keys[i]] for i in idx]) * HEAD
        for si, s in enumerate(shifts):
            Rr = np.roll(Rs[k][:, 1:], -int(s), axis=0)           # day d 的持仓配 day d+s 的收益
            g = np.einsum("cdj,dj->cd", Pst, Rr, optimize=False) - Tst
            m_ = g.mean(1); sd_ = g.std(1, ddof=1)
            plc[idx, si] = np.where(sd_ > 0, m_ / sd_ * np.sqrt(252), np.nan)
        print(f"  安慰剂 {k}min 完成 {time.time()-t0:.0f}s")
    mx = np.nanmax(plc, axis=0)
    df["sr_obs"] = obs
    df["p_single"] = [(1 + np.sum(plc[i] >= obs[i])) / (1 + NPERM) if np.isfinite(obs[i]) else np.nan for i in range(len(keys))]
    df["p_family"] = [(1 + np.sum(mx >= obs[i])) / (1 + NPERM) if np.isfinite(obs[i]) else np.nan for i in range(len(keys))]  # 恒空仓配置(Sharpe 无定义)不参与
    df["plc_mean"] = np.nanmean(plc, axis=1)
    df["pass_i"] = (df.sr_dev > 0) & (df.sr_test > 0); df["pass_ii"] = df.breakeven >= 2.0; df["pass_iii"] = df.p_family < 0.05
    df["PASS"] = df.pass_i & df.pass_ii & df.pass_iii
    df.to_csv(TAB / "intraday_tech.csv", index=False)
    pd.to_pickle({"df": df, "plc_max": mx, "shifts": shifts}, OUT / "intraday_tech.pkl")
    report(df, mx)
    print(f"总耗时 {time.time()-t0:.0f}s")


def report(df, mx):
    pd.set_option("display.width", 250, "display.max_rows", 400, "display.float_format", lambda x: f"{x:8.3f}")
    print("\n===== 汇总(净 Sharpe@1bp / 安慰剂) =====")
    print(f"配置数 {len(df)};PASS = {int(df.PASS.sum())};族内 max 安慰剂 Sharpe 分位:50%={np.nanpercentile(mx,50):.2f} 95%={np.nanpercentile(mx,95):.2f}")
    print(f"净 Sharpe@1bp>0 的配置:{int((df['sr@1.0']>0).sum())}/{len(df)};dev&test 同正:{int(df.pass_i.sum())};breakeven≥2bp:{int(df.pass_ii.sum())};p_family<0.05:{int(df.pass_iii.sum())}")
    print("\n-- 按模式 × 周期:中位数 --")
    print(df.groupby(["k", "mode"])[["rt_day", "gross_bp", "net_bp@1.0", "sr_gross", "sr@1.0", "breakeven"]].median())
    cols = ["k", "ind", "mode", "rt_day", "gross_bp", "net_bp@1.0", "sr_gross", "sr@0.25", "sr@1.0", "breakeven", "sr_dev", "sr_test", "p_single", "p_family"]
    print("\n-- MACD(全部模式)--"); print(df[df.ind == "MACD"][cols].to_string(index=False))
    print("\n-- 毛 Sharpe 最高 15 个 --"); print(df.sort_values("sr_gross", ascending=False).head(15)[cols].to_string(index=False))
    print("\n-- 净 Sharpe@1bp 最高 15 个 --"); print(df.sort_values("sr@1.0", ascending=False).head(15)[cols].to_string(index=False))
    print("\n-- 盈亏平衡成本最高 10 个(毛收益>0)--"); print(df[df.gross_bp > 0].sort_values("breakeven", ascending=False).head(10)[cols].to_string(index=False))
    print("\n-- 研报口径(LO)全部 32 指标 × 5/15 分钟 --")
    print(df[df["mode"] == "LO"].sort_values(["k", "sr@1.0"], ascending=[True, False])[cols].to_string(index=False))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "run"
    if arg == "selftest": selftest(*load_full()[0:1], *load_full()[1:])
    else: run()
