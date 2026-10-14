"""② 杠杆 ETF 收盘再平衡流:增厚利润(H2)还是优化执行(H3/H4)?—— docs/PROPOSAL_letf_rebalance.md(预注册)。
python3 letf_rebalance_study.py"""
import os, sys, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, statsmodels.api as sm
import liquidity_inverse_study as LI, lowvol_avoidance_study as L, sota_stack_study as S, round3_common as C
from etft.intraday.bars import load_bars
import paths as _P

OUT, TAB = L.OUT, L.TAB
BETA = {"TQQQ": 3, "SQQQ": -3, "QLD": 2, "QID": -2}
NPERM = 2000


def load_aum():
    out = {}
    for t in BETA:
        d = pd.read_csv(_P.RAW / "letf" / f"{t}.csv"); d["Date"] = pd.to_datetime(d.Date, format="%m/%d/%Y")
        a = pd.to_numeric(d.set_index("Date")["Assets Under Management"], errors="coerce").sort_index()
        out[t] = a[~a.index.duplicated()]
    return pd.DataFrame(out)


def hac(y, X, lags=5):
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags}); return m


def main():
    pd.set_option("display.width", 260, "display.max_rows", 200, "display.float_format", lambda x: f"{x:9.4f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year; cols = list(b.c.columns); ix = {t: cols.index(t) for t in ("15:30", "15:31", "15:45", "15:46", "16:00")}
    full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); c, o = b.c.to_numpy(), b.o.to_numpy(); prev = b.prev_close_adj.to_numpy(); close = c[:, ix["16:00"]]
    AUM = load_aum(); aum = AUM.reindex(days.union(AUM.index)).ffill().shift(1).reindex(days)          # t−1 的 AUM(PIT)
    q = pd.read_parquet(_P.RAW / "yahoo" / "QQQ.parquet"); q.index = pd.to_datetime(q.index).tz_localize(None)
    adv = (q.close_raw * q.volume).rolling(20).mean().shift(1).reindex(days)
    gearing = sum(aum[t] * BETA[t] * (BETA[t] - 1) for t in BETA)           # Σ AUM·β(β−1)(美元)
    A = gearing / adv                                                        # LETF 相对规模
    print(f"Σ AUM·β(β−1) 中位 ${float(gearing.median())/1e9:.1f}B(2019 中位 ${float(gearing[yrs==2019].median())/1e9:.1f}B → 2026 ${float(gearing[yrs==2026].median())/1e9:.1f}B);"
          f"QQQ 20 日均成交额中位 ${float(adv.median())/1e9:.1f}B;A = 两者之比 中位 {float(A.median()):.3f}")
    print("各基金 β(β−1)·AUM 占比(2026 中位):", {t: f"{float((aum[t]*BETA[t]*(BETA[t]-1)/gearing)[yrs==2026].median()):.1%}" for t in BETA})
    rows, D = [], {}
    for tau in ("15:30", "15:45"):
        k = ix[tau]; fill = ix["15:31"] if tau == "15:30" else ix["15:46"]
        r_rest = pd.Series(c[:, k] / prev - 1, index=days); r_last = pd.Series(close / c[:, k] - 1, index=days); r_trade = pd.Series(close / o[:, fill] - 1, index=days)
        Dd = gearing * r_rest; Dn = Dd / adv
        last_dollar = pd.Series(b.v.to_numpy()[:, k + 1:].sum(1) * close, index=days)        # τ 之后到收盘的 QQQ 成交额
        D[tau] = dict(r_rest=r_rest, r_last=r_last, r_trade=r_trade, Dn=Dn, Dd=Dd, ratio_last=Dd.abs() / last_dollar)
        m = full & r_rest.notna() & A.notna() & (days >= "2019-01-01")
        print(f"\n[{tau}] |需求| 中位 ${float(Dd[m].abs().median())/1e9:.2f}B;占 τ 之后 QQQ 成交额 中位 {float(D[tau]['ratio_last'][m].median()):.1%}(90 分位 {float(D[tau]['ratio_last'][m].quantile(.9)):.1%})")
        # ---- H1 回归
        y = r_last[m] * 1e4; x1 = r_rest[m] * 1e4; Ac = (A[m] - A[m].mean()) / A[m].std()
        m1 = hac(y, pd.DataFrame({"r_rest": x1})); m2 = hac(y, pd.DataFrame({"r_rest": x1, "r_rest×A": x1 * Ac, "A": Ac}))
        rows.append(dict(τ=tau, 模型="只有 r_rest", b=m1.params["r_rest"], t_b=m1.tvalues["r_rest"], c=np.nan, t_c=np.nan, R2=m1.rsquared, n=int(m.sum())))
        rows.append(dict(τ=tau, 模型="+ r_rest×A(LETF 规模)", b=m2.params["r_rest"], t_b=m2.tvalues["r_rest"], c=m2.params["r_rest×A"], t_c=m2.tvalues["r_rest×A"], R2=m2.rsquared, n=int(m.sum())))
        by = {}
        for yv in sorted(set(yrs[m.to_numpy()])):
            mm = m & (yrs == yv); r_ = hac(r_last[mm] * 1e4, pd.DataFrame({"r_rest": r_rest[mm] * 1e4})); by[int(yv)] = (round(float(r_.params["r_rest"]), 4), round(float(r_.tvalues["r_rest"]), 2))
        print(f"  分年 b(t):{by}")
        qv = pd.qcut(r_rest[m].abs().rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        print("  按 |r_rest| 五分位:sign(r_rest)·r_last 均值(bp):", (np.sign(r_rest[m]) * r_last[m] * 1e4).groupby(qv).mean().round(2).to_dict(),
              "| |r_rest| 中位(%):", (r_rest[m].abs() * 100).groupby(qv).median().round(2).to_dict())
    H1 = pd.DataFrame(rows); print("\nH1) 末段收益 ~ 前段收益(HAC,单位 bp):"); print(H1.to_string(index=False)); H1.to_csv(TAB / "letf_h1.csv", index=False)
    # ==================================================== H2 增厚利润:末段 overlay
    print("\n" + "=" * 100 + "\nH2) 增厚利润:τ 按 sign(D) 做 QQQ,MOC 平仓,成本 1bp/边\n" + "=" * 100)
    base_mask = (full & (days >= "2019-01-01")).to_numpy(); rg = np.random.default_rng(11); rows, nulls, obs, series = [], {}, {}, {}
    for tau in ("15:30", "15:45"):
        Dn, rt = D[tau]["Dn"], D[tau]["r_trade"]
        thr_src = Dn.abs().where(pd.Series(base_mask, index=days))
        for qq in (0.0, 0.5, 0.7):
            thr = thr_src.rolling(252, min_periods=60).quantile(qq).shift(1) if qq > 0 else pd.Series(0.0, index=days)
            trade = base_mask & (Dn.abs() > thr).to_numpy() & Dn.notna().to_numpy()
            sg = np.sign(Dn.to_numpy()); gross = np.where(trade, sg * rt.to_numpy(), 0.0); net = np.where(trade, gross - 2e-4, 0.0)
            x = pd.Series(net, index=days)[base_mask]; key = (tau, qq); series[key] = pd.Series(net, index=days)
            o_ = L.sr(x); obs[key] = o_
            tr_i = np.where(trade[base_mask])[0]; rv = (np.abs(gross) * np.sign(gross))[base_mask]
            nl = np.empty(NPERM)
            for i in range(NPERM):
                s_ = np.zeros(base_mask.sum()); fl = rg.choice([-1.0, 1.0], len(tr_i))
                s_[tr_i] = fl * np.abs(rt.to_numpy()[base_mask][tr_i]) * np.sign(rt.to_numpy()[base_mask][tr_i]) - 2e-4
                nl[i] = L.sr(pd.Series(s_))
            nulls[key] = nl
            t_ = x[x != 0]
            rows.append(dict(τ=tau, 阈值q=qq, 交易天数=int(trade.sum()), 每笔毛bp=1e4 * float(gross[trade].mean()), 每笔净bp=1e4 * float(t_.mean()), 胜率=float((t_ > 0).mean()),
                             Sharpe=o_, dev_每笔净bp=1e4 * float(x[(x.index.year <= 2022) & (x != 0)].mean()), test_每笔净bp=1e4 * float(x[(x.index.year >= 2023) & (x != 0)].mean()),
                             年化bp=1e4 * 252 * float(x.mean()), 盈亏平衡成本bp每边=1e4 * float(gross[trade].mean()) / 2, p单=float(np.mean(nl >= o_))))
    fam = np.max(np.vstack(list(nulls.values())), axis=0); H2 = pd.DataFrame(rows); H2["p族内(6)"] = [float(np.mean(fam >= obs[(a, q_)])) for a, q_ in zip(H2.τ, H2.阈值q)]
    print(H2.to_string(index=False)); H2.to_csv(TAB / "letf_h2_overlay.csv", index=False)
    # ==================================================== H3 优化执行:Sleeve 的 MOC
    print("\n" + "=" * 100 + "\nH3) 优化执行:Sleeve 在 15:45 仍持有的仓位,持到 MOC vs 15:46 提前平仓\n" + "=" * 100)
    PH = LI.run_phases_fills(b); TR = pd.concat([LI.trades_from_fills(F, off) for off, (rr, F) in PH.items() if len(F)], ignore_index=True)
    TR = TR[(TR.t_in <= "15:45") & (TR.t_out > "15:45")].copy()
    o1546 = pd.Series(o[:, ix["15:46"]], index=days); TR["G"] = TR.shares * (TR.exit_px - TR.day.map(o1546))
    print(f"  15:45 仍持有的成交(30 相位合计){len(TR)} 笔;其中 MOC 平仓 {float((TR.t_out=='16:00').mean()):.1%}、15:46 之后止损 {float((TR.t_out!='16:00').mean()):.1%}")
    gpd = TR.groupby(["day", "phase"]).agg(G=("G", "sum"), pos=("shares", "sum")).reset_index()
    sgD = np.sign(D["15:45"]["Dn"]); gpd["aligned"] = (np.sign(gpd.pos).to_numpy() == gpd.day.map(sgD).to_numpy())
    absD = D["15:45"]["Dn"].abs(); medD = absD.where(pd.Series(base_mask, index=days)).rolling(252, min_periods=60).median().shift(1); gpd["bigD"] = (gpd.day.map(absD) >= gpd.day.map(medD)).to_numpy()
    base = pd.read_pickle(OUT / "sleeve_anatomy_curve.pkl"); base.index = pd.to_datetime(base.index); base = base.reindex(days).fillna(0.0)
    td = (days.dayofweek != 0) & base_mask
    def policy(drop):
        g = gpd[drop].groupby("day").G.sum() / 30.0; return base - g.reindex(days).fillna(0.0)
    al = gpd.aligned; print(f"  持仓与 LETF 流同向的比例 {float(al.mean()):.1%};每笔 G(bp,资本口径):同向 {1e4*gpd.G[al].mean():+.2f}、反向 {1e4*gpd.G[~al].mean():+.2f}")
    print(f"  大流量日(|Dn|≥中位)同向 {1e4*gpd.G[al & gpd.bigD].mean():+.2f} / 反向 {1e4*gpd.G[~al & gpd.bigD].mean():+.2f};小流量日同向 {1e4*gpd.G[al & ~gpd.bigD].mean():+.2f} / 反向 {1e4*gpd.G[~al & ~gpd.bigD].mean():+.2f}")
    dayG = gpd.groupby("day").G.sum() / 30.0; dG = dayG.reindex(days).fillna(0.0)[td]
    tG = sm.OLS(dG.to_numpy() * 1e4, np.ones(len(dG))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    print(f"  Sleeve 日均从 15:46 到收盘赚到的 G = {1e4*dG.mean():+.3f}bp/日(HAC t={tG.tvalues[0]:.2f});占 Sleeve 日均收益 {dG.mean()/base[td].mean():.1%}")
    pols = {"E0 现行(全部 MOC)": np.zeros(len(gpd), bool), "E1 一律 15:46 平仓": np.ones(len(gpd), bool), "E2 反向于流则提前平": (~gpd.aligned).to_numpy(),
            "E3 仅大流量日反向才提前平": ((~gpd.aligned) & gpd.bigD).to_numpy()}
    rows = []; obs3 = {}
    for nm, dr in pols.items():
        x = policy(dr); rows.append(dict(方案=nm, Sleeve_Sharpe=L.sr(x[td]), ΔSR=L.sr(x[td]) - L.sr(base[td]), dev=L.sr(x[td & (yrs <= 2022)]) - L.sr(base[td & (yrs <= 2022)]),
                                     test=L.sr(x[td & (yrs >= 2023)]) - L.sr(base[td & (yrs >= 2023)]), 日均变化bp=1e4 * float((x - base)[td].mean()))); obs3[nm] = rows[-1]["ΔSR"]
    # 置换:随机翻转 sign(D)(按日)
    ud = gpd.day.unique(); nl3 = {k: np.empty(NPERM) for k in ("E2 反向于流则提前平", "E3 仅大流量日反向才提前平")}
    posd = np.sign(gpd.pos).to_numpy(); dsg = gpd.day.map(sgD).to_numpy(); big = gpd.bigD.to_numpy(); day_codes = pd.Categorical(gpd.day, categories=ud).codes
    for i in range(NPERM):
        fl = rg.choice([-1.0, 1.0], len(ud))[day_codes]; alig = posd == dsg * fl
        for k_, dr in (("E2 反向于流则提前平", ~alig), ("E3 仅大流量日反向才提前平", (~alig) & big)):
            nl3[k_][i] = L.sr(policy(dr)[td]) - L.sr(base[td])
    fam3 = np.max(np.vstack(list(nl3.values())), axis=0); H3 = pd.DataFrame(rows)
    H3["p族内"] = [float(np.mean(fam3 >= obs3[n])) if n in nl3 else np.nan for n in H3.方案]
    print(H3.to_string(index=False)); H3.to_csv(TAB / "letf_h3_exec.csv", index=False)
    # ==================================================== H4 QQQ 腿调仓时点
    print("\n" + "=" * 100 + "\nH4) 优化执行:QQQ 腿的调仓改在 15:46(而非收盘)\n" + "=" * 100)
    cand, _, _ = S.build_candidates(b); px = C.load_close("yahoo/QQQ.parquet"); rr = px.pct_change().fillna(0.0)
    A2C = pd.read_pickle(OUT / "regime_timing_r2.pkl")["E2"]["A2_C|e0.5|alone"]; W = S.bh_weights(px, rr, {"FFA-SD": cand["FFA-SD"]}, A2C)
    rtr = D["15:45"]["r_trade"]
    for nm, key in (("现用 RV21", ("RV21[0.5,1.5]", "无趋势")), ("① RV21+SMA200", ("RV21[0.5,1.5]", "SMA200|e=0.5")), ("② VT[FFA-SD]+SMA200", ("VT[FFA-SD,cap1.5]", "SMA200|e=0.5"))):
        w = W[key]; dw = w.diff().shift(-1)            # t 日收盘执行的调仓 = w_{t+1} − w_t
        x = (dw.reindex(days) * rtr)[base_mask & rtr.notna().to_numpy()]
        print(f"  {nm}: 年换手 {float(w.diff().abs()['2019':].sum()/7.75):.1f}×;改在 15:46 执行的年化改善 {1e4*252*float(x.mean()):+.2f}bp(上界:15:45 时信号尚未完全确定)")
    pd.to_pickle(dict(D=D, series=series), OUT / "letf_rebalance.pkl")


if __name__ == "__main__":
    main()
