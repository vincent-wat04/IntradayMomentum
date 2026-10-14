"""① 流动性/换手(成本真实性、容量、作为信号) ② 买 SQQQ vs 空 TQQQ + 恐慌时仓位不对称
—— docs/PROPOSAL_liquidity_and_inverse.md(预注册)。python3 liquidity_inverse_study.py"""
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
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowvol_avoidance_study as L
import round3_common as C
import sleeve_anatomy as SA
import sota_stack_study as S
import trend_rv_sleeve_study as T
from audit import reimpl_noise as RI
from etft.intraday.bars import load_bars
from etft.intraday.strategy import rel_volume
from sens_rv_weights import pct_w
import paths as _P

OUT, TAB = L.OUT, L.TAB
HS0, CM0 = 0.005, 0.0045          # 现用:固定半价差 + 佣金(美元/股)
NPERM = 2000
PHASES = range(30)


# ======================================================== 数据
def etf_daily():
    import yfinance as yf
    d = yf.download(["TQQQ", "SQQQ", "QQQ"], start="2018-10-01", end="2026-10-12", auto_adjust=True, progress=False, threads=False)
    out = {}
    for t in ("TQQQ", "SQQQ", "QQQ"):
        x = pd.DataFrame({k.lower(): d[k][t] for k in ("Open", "High", "Low", "Close", "Volume")}).dropna()
        x.index = pd.to_datetime(x.index).tz_localize(None); out[t] = x
    return out


def tqqq_traded_px(days):
    """还原 TQQQ 的历史实际成交价(未调整价 × 累积拆股因子),与 sleeve_anatomy.run_phases 的 scale 同口径。
    yfinance 的 Close 始终已按拆股调整,乘回累积因子才得到当时真实的每股价格 —— 按股计费的成本必须用它。"""
    import yfinance as yf
    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-12", auto_adjust=False, progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt; pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None)
    fsp = pd.Series(1.0, index=pt.index)
    for d, r in sp.items(): fsp[pt.index < d] *= r
    return (pt * fsp).reindex(days).ffill()


def corwin_schultz(h, l, smooth=21):
    """CS(2012) 有效价差比例估计;负值截 0,再取滚动中位数平滑。"""
    lh, ll = np.log(h), np.log(l)
    b = (lh - ll) ** 2 + ((lh - ll) ** 2).shift(1)                       # 两日各自的 (lnH-lnL)^2 之和
    h2 = np.maximum(h, h.shift(1)); l2 = np.minimum(l, l.shift(1))
    g = (np.log(h2 / l2)) ** 2
    k = 3 - 2 * np.sqrt(2)
    a = (np.sqrt(2 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
    s = 2 * (np.exp(a) - 1) / (1 + np.exp(a))
    return s.clip(lower=0).rolling(smooth, min_periods=5).median()


# ======================================================== Sleeve:成交明细 + 逐笔重建
def run_phases_fills(b):
    """30 相位:返回 (rr_sim[off], F[off]) —— rr_sim 是 simulate 原始日收益(含固定成本)。"""
    p = OUT / "liq_phase_fills.pkl"
    if p.exists(): return pd.read_pickle(p)
    out = {}
    for off in PHASES:
        rr, F = RI.simulate(b, checks=SA.chk(off), sizing="voltarget", max_lev=4.0, reentry="hold", half_spread=HS0, commission=CM0, **SA.BASE)
        F = F.copy()
        if len(F): F["day"] = pd.to_datetime(F.day)
        out[off] = (rr, F)
        print(f"  相位 {off:2d} 完成", flush=True) if off % 10 == 0 else None
    pd.to_pickle(out, p); return out


def trades_from_fills(F, off):
    """把成交明细配对成逐笔:返回 DataFrame(day, side, shares, entry_px, exit_px, cm_in, cm_out, pnl_net, k_in, k_out)。"""
    rows = []
    for d, g in F.groupby("day", sort=False):
        cur = None
        for _, r in g.iterrows():
            kind = str(r.kind)
            if kind.startswith("ENTRY"): cur = r
            elif cur is not None:
                sh = float(cur.shares)
                rows.append(dict(day=d, side=int(np.sign(sh)), shares=sh, entry_px=float(cur.price), exit_px=float(r.price),
                                 cm_in=float(cur.commission), cm_out=float(r.commission), t_in=str(cur.clock), t_out=str(r.clock),
                                 pnl_gross=sh * (float(r.price) - float(cur.price)), phase=off))
                cur = None
    T_ = pd.DataFrame(rows)
    if len(T_): T_["pnl_net"] = T_.pnl_gross - T_.cm_in - T_.cm_out
    return T_


def main():
    t0 = time.time(); pd.set_option("display.width", 270, "display.max_rows", 400, "display.float_format", lambda x: f"{x:9.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year; td = pd.Series(days.dayofweek != 0, index=days)
    base = pd.read_pickle(OUT / "sleeve_anatomy_curve.pkl"); base.index = pd.to_datetime(base.index); base = base.reindex(days).fillna(0.0)
    E = etf_daily(); tq, sq, qq = E["TQQQ"], E["SQQQ"], E["QQQ"]
    px_tq = tqqq_traded_px(days)                                           # TQQQ 历史实际成交价(按股计费必须用它)
    scale = (b.day_close / px_tq) / 3.0                                    # 与 sleeve_anatomy.run_phases 完全同口径
    print(f"数据:QQQ 分钟 {len(days)} 日;TQQQ/SQQQ 日频 {tq.index[0].date()}~{tq.index[-1].date()};scale 中位 {float(scale.median()):.2f}")
    PH = run_phases_fills(b); print(f"30 相位成交明细就绪 {time.time()-t0:.0f}s")
    TR = pd.concat([trades_from_fills(F, off) for off, (rr, F) in PH.items() if len(F)], ignore_index=True)
    offs = list(PH)
    cm_day = pd.concat([PH[o][1].groupby("day").commission.sum().reindex(days).fillna(0.0) for o in offs], axis=1).astype(float); cm_day.columns = offs
    rr_sim = pd.concat([PH[o][0].reindex(days).fillna(0.0) for o in offs], axis=1).astype(float); rr_sim.columns = offs
    base_chk = (rr_sim + cm_day - cm_day.mul(scale.fillna(1.0), axis=0)).mean(axis=1)
    print(f"[校验1] 用成交明细重算的基线 vs sleeve_anatomy_curve 最大绝对差 {float((base_chk - base).abs().max())*1e4:.6f} bp")
    # 逐笔重建(用于腿级缩放):每相位每日 Σpnl_net 应等于 rr_sim
    pv = TR.pivot_table(index="day", columns="phase", values="pnl_net", aggfunc="sum").reindex(days).fillna(0.0)
    print(f"[校验2] 逐笔净损益聚合 vs simulate 原始日收益 最大绝对差 {float((pv.reindex(columns=offs) - rr_sim).abs().max().max())*1e4:.6f} bp")
    # ==================================================== L1 成本真实性
    cs = corwin_schultz(tq.high, tq.low).reindex(days).ffill()
    hs_t = (cs * px_tq / 2.0)        # CS 给出的是比例价差,乘实际股价得美元半价差
    n_tq = (cm_day / (HS0 + CM0)).mul(scale.fillna(1.0), axis=0)           # 当日 TQQQ 成交股数(每相位)
    print("\n" + "=" * 110 + "\nL1) 成本真实性(无盘口数据 → 只能做压力测试,不能做估计)\n" + "=" * 110)
    print(f"[方法性否定结果] Corwin–Schultz 对 3x ETF 不可用:比例价差中位 {float(cs.median())*1e4:.1f}bp(>50% 的估计为负被截 0),"
          f"但 80 分位给出半价差 ${float(hs_t[hs_t >= hs_t.quantile(.8)].median()):.3f}/股 ≈ {1e4*float((hs_t/px_tq)[hs_t >= hs_t.quantile(.8)].median()):.0f}bp,"
          f"而 TQQQ 实际盘口约 1 个最小跳动($0.01 全价差 ≈ {1e4*0.005/float(px_tq.median()):.1f}bp 半价差)。CS 把杠杆 ETF 的巨大日内波幅误判为价差 ⇒ 弃用。")
    sig_tq = (np.log(tq.high / tq.low)).reindex(days).ffill(); sig_rel = (sig_tq / sig_tq.rolling(252, min_periods=60).median()).clip(upper=10)
    def recost(hs_series):
        return (rr_sim + cm_day - n_tq.mul(hs_series.reindex(days).fillna(HS0) + CM0, axis=0)).mean(axis=1)
    rows = []
    for nm, hs in [(f"现用 固定 ${HS0}", pd.Series(HS0, index=days))] + \
                  [(f"固定 {m:g}× = ${HS0*m:.3f}", pd.Series(HS0 * m, index=days)) for m in (2, 3, 5, 10)] + \
                  [(f"随波动放大 a={a_:g}", HS0 * (1 + a_ * (sig_rel - 1)).clip(lower=1.0)) for a_ in (1, 2)]:
        x = recost(hs)
        rows.append(dict(价差设定=nm, 半价差中位=float(hs.median()), 等价半价差bp=1e4 * float((hs / px_tq).median()), Sharpe_全=L.sr(x[td.to_numpy()]),
                         Sharpe_dev=L.sr(x[td.to_numpy() & (yrs <= 2022)]), Sharpe_test=L.sr(x[td.to_numpy() & (yrs >= 2023)]),
                         均值bp=1e4 * x[td.to_numpy()].mean(), Sharpe_2020=L.sr(x[td.to_numpy() & (yrs == 2020)])))
    L1 = pd.DataFrame(rows); print(L1.to_string(index=False)); L1.to_csv(TAB / "liq_cost_realism.csv", index=False)
    lo_, hi_ = 1.0, 60.0
    for _ in range(40):
        mid = (lo_ + hi_) / 2
        if recost(pd.Series(HS0 * mid, index=days))[td.to_numpy()].mean() > 0: lo_ = mid
        else: hi_ = mid
    print(f"盈亏平衡半价差(全期日均净收益=0):{(lo_+hi_)/2:.1f}× 现用 = ${HS0*(lo_+hi_)/2:.3f}/股 ≈ {1e4*HS0*(lo_+hi_)/2/float(px_tq.median()):.1f}bp 半价差"
          f"(即 TQQQ 全价差需达到约 {2*HS0*(lo_+hi_)/2/float(px_tq.median())*1e4:.0f}bp ≈ ${2*HS0*(lo_+hi_)/2:.2f}/股 才吃光利润)")
    # 容量
    adv = (tq.close * tq.volume).rolling(20).mean().reindex(days).ffill()
    notional = n_tq.mean(axis=1) * px_tq          # 每单位资本的 TQQQ 日成交额
    rows = []
    for part in (0.01, 0.05, 0.10):
        capm = (part * adv / notional.replace(0, np.nan))
        rows.append(dict(参与率=f"{part:.0%}", 最大资本_中位=float(capm.median()), 最大资本_10分位=float(capm.quantile(.1)),
                         恐慌日中位=float(capm[(yrs >= 2019) & (hs_t >= hs_t.quantile(.8)).to_numpy()].median())))
    CAP = pd.DataFrame(rows); print("\n容量(最大资本,美元;TQQQ 20 日 ADV 中位 ${:.1f}亿)".format(float(adv.median()) / 1e8)); print(CAP.to_string(index=False)); CAP.to_csv(TAB / "liq_capacity.csv", index=False)
    print("  每单位资本的 TQQQ 日成交额(中位):", f"{float(notional.median()):.2f}×资本")
    # ==================================================== I1 工具楔子
    print("\n" + "=" * 110 + "\nI1) 买 SQQQ vs 空 TQQQ(开盘→收盘日频会计;描述性)\n" + "=" * 110)
    oc = {k: (v.close / v.open - 1) for k, v in E.items()}
    W = (oc["TQQQ"] + oc["SQQQ"]).dropna(); vix = pd.read_parquet(_P.RAW / "cboe" / "VIX.parquet")["close"]
    vix.index = pd.to_datetime(vix.index).tz_localize(None); vix = vix[~vix.index.duplicated()]
    dlv = np.log(vix).diff(); rq = oc["QQQ"]
    panic_d = ((dlv.shift(1) > 0) & (rq.shift(1) < 0)).reindex(W.index).fillna(False)
    rows = []
    # 注:楔子与融券费都按 ETF 名义计价,故盈亏平衡率不再除以 3
    for nm, m in (("全样本", pd.Series(True, index=W.index)), ("恐慌日 P1", panic_d), ("非恐慌", ~panic_d),
                  ("|QQQ日内|最高5%", W.index.isin(rq.reindex(W.index).abs().nlargest(int(.05 * len(W))).index)), ("VIX>30", (vix.shift(1).reindex(W.index) > 30).fillna(False))):
        x = W[np.asarray(m)]
        rows.append(dict(样本=nm, n=len(x), 楔子均值bp=1e4 * x.mean(), 中位bp=1e4 * x.median(), std_bp=1e4 * x.std(),
                         单位敞口bp=1e4 * x.mean() / 3, 融券盈亏平衡率年化=-x.mean() * 252))
    I1 = pd.DataFrame(rows); print(I1.to_string(index=False)); I1.to_csv(TAB / "inv_wedge.csv", index=False)
    print("  分年 融券盈亏平衡率(年化):", {int(y): f"{-g.mean()*252:.1%}" for y, g in W.groupby(W.index.year)})
    # 持仓时长折算
    ts = TR[TR.side < 0].copy()
    hold = (pd.to_datetime(ts.t_out, format="%H:%M") - pd.to_datetime(ts.t_in, format="%H:%M")).dt.total_seconds() / 3600
    print(f"  Sleeve 空头笔数 {len(ts)}(30 相位),平均持仓 {float(hold.mean()):.2f} 小时(交易日 6.5 小时);"
          f"按时长折算的每笔工具楔子 ≈ {1e4*W.mean()*float(hold.mean())/6.5:.2f}bp(ETF 价格口径)")
    dv = pd.DataFrame({"TQQQ": tq.close * tq.volume, "SQQQ": sq.close * sq.volume}).dropna()
    print(f"  成交额(百万美元,中位):TQQQ {float(dv.TQQQ.median())/1e6:.0f} / SQQQ {float(dv.SQQQ.median())/1e6:.0f};"
          f"恐慌日 TQQQ {float(dv.TQQQ[panic_d.reindex(dv.index).fillna(False)].median())/1e6:.0f} / SQQQ {float(dv.SQQQ[panic_d.reindex(dv.index).fillna(False)].median())/1e6:.0f};"
          f"SQQQ/TQQQ 成交额比 全样本 {float((dv.SQQQ/dv.TQQQ).median()):.2f} / 恐慌日 {float((dv.SQQQ/dv.TQQQ)[panic_d.reindex(dv.index).fillna(False)].median()):.2f}")
    # ==================================================== I2 恐慌时仓位不对称
    print("\n" + "=" * 110 + "\nI2) 恐慌时仓位不对称(腿级精确缩放)\n" + "=" * 110)
    c = b.c.to_numpy(); O = b.day_open.to_numpy(); gap = pd.Series(np.log(O / b.prev_close_adj.to_numpy()), index=days)
    r30 = pd.Series(np.log(c[:, 29] / O), index=days); vixd = vix.shift(1).reindex(days).ffill(); dlvd = dlv.shift(1).reindex(days).ffill()
    rqd = pd.Series(b.day_close.to_numpy() / b.prev_close_adj.to_numpy() - 1, index=days).shift(1)
    P = {"P1 VIX↑&跌": (dlvd > 0) & (rqd < 0), "P2 P1&VIX>25": (dlvd > 0) & (rqd < 0) & (vixd > 25),
         "P3 跳空跌&VIX>20": (gap < 0) & (vixd > 20), "P4 P3&首半小时跌": (gap < 0) & (vixd > 20) & (r30 < 0)}
    P = {k: (v.fillna(False) & td) for k, v in P.items()}
    sel = td.to_numpy() & (yrs >= 2019)
    # 腿级日收益
    legs = {}
    for sd_, nm in ((1, "long"), (-1, "short")):
        legs[nm] = TR[TR.side == sd_].pivot_table(index="day", columns="phase", values="pnl_net", aggfunc="sum").reindex(days).fillna(0.0).mean(axis=1)
    cost_adj = (cm_day - cm_day.mul(scale.fillna(1.0), axis=0)).mean(axis=1)    # TQQQ 成本换算(两腿共享,按成交额分摊)
    sh_leg = {nm: TR[TR.side == (1 if nm == "long" else -1)].assign(cm=lambda z: z.cm_in + z.cm_out).pivot_table(index="day", columns="phase", values="cm", aggfunc="sum").reindex(days).fillna(0.0).mean(axis=1) for nm in ("long", "short")}
    tot_cm = sh_leg["long"] + sh_leg["short"]
    adj_leg = {nm: (cost_adj * (sh_leg[nm] / tot_cm.replace(0, np.nan))).fillna(0.0) for nm in ("long", "short")}
    L_ = legs["long"] + adj_leg["long"]; S_ = legs["short"] + adj_leg["short"]
    print(f"[校验3] 腿级(多+空)重建 vs 基线 最大绝对差 {float((L_ + S_ - base).abs().max())*1e4:.6f} bp")
    print(f"全期(周二至周五,2019+):多头腿 {1e4*L_[sel].mean():+.2f}bp/日、空头腿 {1e4*S_[sel].mean():+.2f}bp/日")
    rows = []
    for pn, m in P.items():
        mk = m.to_numpy() & sel; nm_ = ~m.to_numpy() & sel
        rows.append(dict(状态=pn, 恐慌天数=int(mk.sum()), 多头_恐慌bp=1e4 * L_[mk].mean(), 空头_恐慌bp=1e4 * S_[mk].mean(),
                         多头_平时bp=1e4 * L_[nm_].mean(), 空头_平时bp=1e4 * S_[nm_].mean(), 合计_恐慌bp=1e4 * base[mk].mean(), 合计_平时bp=1e4 * base[nm_].mean(),
                         空头笔数占比=float(TR[TR.day.isin(days[mk])].side.lt(0).mean()) if mk.sum() else np.nan))
    DS = pd.DataFrame(rows); print("\n腿级描述(bp/日):"); print(DS.to_string(index=False)); DS.to_csv(TAB / "inv_leg_desc.csv", index=False)
    # 配置族
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    port = lambda s_: T.build(bh, s_.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px_all, p_all)[0]
    base_p = port(base); rg = np.random.default_rng(11); idx = np.where(sel)[0]; bsr = L.sr(base.iloc[idx])
    KS = [(1, 1.5), (1, 2), (0.5, 1.5), (0, 1.5), (1, 3)]
    pool_by_dow = {w: np.where(sel & (days.dayofweek == w))[0] for w in (1, 2, 3, 4)}
    rows, nulls, obs = [], {}, {}
    for pn, m in P.items():
        mv = m.to_numpy(); cnt = pd.Series(days[mv & sel].dayofweek).value_counts()
        for kl, ks in KS:
            new = base + (kl - 1) * L_.where(pd.Series(mv, index=days), 0.0) + (ks - 1) * S_.where(pd.Series(mv, index=days), 0.0)
            o = L.sr(new.iloc[idx]) - bsr; key = (pn, kl, ks); obs[key] = o
            nl = np.empty(NPERM)
            for i in range(NPERM):
                pm = np.zeros(len(days), bool)
                for w, nn in cnt.items():
                    pl = pool_by_dow.get(w, np.array([], int))
                    if len(pl): pm[rg.choice(pl, min(nn, len(pl)), replace=False)] = True
                pms = pd.Series(pm, index=days)
                x = base + (kl - 1) * L_.where(pms, 0.0) + (ks - 1) * S_.where(pms, 0.0)
                nl[i] = L.sr(x.iloc[idx]) - bsr
            nulls[key] = nl
            xp = port(new); pw = lambda z, k: (z[(z.index >= "2019-01-01") & (z.index <= "2022-12-31")] if k == "dev" else z[z.index >= "2023-01-01"] if k == "test" else z[z.index >= "2019-01-01"])
            row = dict(状态=pn, k_long=kl, k_short=ks, Sleeve_ΔSR_全=o,
                       Sleeve_ΔSR_dev=L.sr(new[sel & (yrs <= 2022)]) - L.sr(base[sel & (yrs <= 2022)]), Sleeve_ΔSR_test=L.sr(new[sel & (yrs >= 2023)]) - L.sr(base[sel & (yrs >= 2023)]),
                       组合ΔSR_全=L.sr(pw(xp, "全")) - L.sr(pw(base_p, "全")), 组合ΔSR_dev=L.sr(pw(xp, "dev")) - L.sr(pw(base_p, "dev")), 组合ΔSR_test=L.sr(pw(xp, "test")) - L.sr(pw(base_p, "test")))
            pan = S.panel(new[sel]); row.update(CVaR5bp=pan["CVaR5%bp"], MDD=pan["MDD"], 最差日bp=pan["最差日bp"], 峰度=pan["峰度"], p单=float(np.mean(nl >= o)))
            rows.append(row)
    fam = np.max(np.vstack([nulls[k] for k in nulls]), axis=0)
    I2 = pd.DataFrame(rows); I2["p族内(20)"] = [float(np.mean(fam >= obs[(a, b_, c_)])) for a, b_, c_ in zip(I2.状态, I2.k_long, I2.k_short)]
    bp = S.panel(base[sel]); print(f"\n基线:CVaR5% {bp['CVaR5%bp']:.1f}bp、MDD {bp['MDD']:.3f}、最差日 {bp['最差日bp']:.1f}bp、峰度 {bp['峰度']:.1f}")
    print(I2.to_string(index=False)); I2.to_csv(TAB / "inv_asym_sizing.csv", index=False)
    ok = I2[(I2.Sleeve_ΔSR_dev > 0) & (I2.Sleeve_ΔSR_test > 0)]
    print(f"\ndev/test 同号为正:{len(ok)}/{len(I2)};族内 p 最小 {I2['p族内(20)'].min():.3f};全期 ΔSR 最大 {I2.Sleeve_ΔSR_全.max():+.3f}")
    # ==================================================== L2 流动性作为信号
    print("\n" + "=" * 110 + "\nL2) 流动性/换手作为信号\n" + "=" * 110)
    dvq = (qq.close * qq.volume).reindex(days).ffill(); dvt = (tq.close * tq.volume).reindex(days).ffill(); dvs = (sq.close * sq.volume).reindex(days).ffill()
    z = lambda x, n=60: ((x - x.rolling(n, min_periods=20).mean()) / x.rolling(n, min_periods=20).std())
    rmin = np.full_like(c, np.nan); rmin[:, 1:] = np.log(c[:, 1:] / c[:, :-1]); rmin[:, 0] = np.log(c[:, 0] / O)
    v = b.v.to_numpy(); lam = np.full(len(days), np.nan)
    for i in range(len(days)):
        x_ = np.sign(rmin[i]) * np.sqrt(np.maximum(v[i], 0)); y_ = rmin[i]; m_ = np.isfinite(x_) & np.isfinite(y_) & (x_ != 0)
        if m_.sum() > 50: lam[i] = np.polyfit(x_[m_], y_[m_], 1)[0]
    F_ = pd.DataFrame({
        "cs_spread_z": z(cs), "amihud_d": z(pd.Series(np.abs(rqd.to_numpy()), index=days) / dvq.shift(1) * 1e12),
        "kyle_lambda": z(pd.Series(lam, index=days) * 1e6).shift(1), "turnover_z": np.log(qq.volume.reindex(days).ffill() / qq.volume.reindex(days).ffill().rolling(20).mean()).shift(1),
        "lev_demand": z(np.log(dvt / dvq)).shift(1), "inv_tilt": z(np.log(dvs / dvt)).shift(1),
        "vol_prof_surprise": np.log(rel_volume(b, 14, "cum")["10:00"].reindex(days))}, index=days)
    y = base * 1e4; FC = list(F_.columns)
    print("特征缺失占比(2019+):", F_.loc["2019":].isna().mean().round(3).to_dict())
    ic = {c_: spearmanr(F_.loc[sel, c_], y[sel], nan_policy="omit")[0] for c_ in FC}
    print("Spearman IC(特征 vs 当日 Sleeve 净损益,2019+):", {k: round(float(v_), 3) for k, v_ in ic.items()})
    pred = pd.Series(np.nan, index=days)
    for yr in range(2021, 2027):
        tr = (days.year < yr) & F_.notna().all(1).to_numpy() & td.to_numpy(); te = (days.year == yr) & F_.notna().all(1).to_numpy()
        if tr.sum() < 250 or te.sum() == 0: continue
        mu, sdv = F_[tr].mean(), F_[tr].std().replace(0, 1)
        pred[te] = Ridge(alpha=10.0).fit(((F_[tr] - mu) / sdv).to_numpy(), y[tr].to_numpy()).predict(((F_[te] - mu) / sdv).to_numpy())
    oos = sel & pred.notna().to_numpy()
    print(f"岭回归 OOS(2021–26,n={int(oos.sum())}):IC {spearmanr(pred[oos], y[oos])[0]:.3f};按预测五分位的 Sleeve 日均 bp:",
          (y[oos].groupby(pd.qcut(pred[oos].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])).mean()).round(1).to_dict())
    rows, nl2, ob2 = [], {}, {}
    perms = [rg.permutation(len(idx)) for _ in range(1000)]
    cands = {**{c_: F_[c_] for c_ in FC}, "合成(岭回归)": -pred}      # 分数越低→越该跳过;对 pred 取负使方向一致
    for nm, sc in cands.items():
        rk = S.roll_rank(sc.where(td))
        for q in (0.2, 0.3):
            s_ = (~(rk <= q).fillna(False)).astype(float); new = base * s_
            o = L.sr(base.iloc[idx] * s_.to_numpy()[idx]) - bsr; key = (nm, q); ob2[key] = o
            nl2[key] = np.array([L.sr(pd.Series(base.to_numpy()[idx] * s_.to_numpy()[idx][p_])) for p_ in perms]) - bsr
            xp = port(new)
            rows.append(dict(特征=nm, q=q, 跳过比例=float((s_[sel] == 0).mean()), Sleeve_ΔSR_全=o,
                             dev=L.sr(new[sel & (yrs <= 2022)]) - L.sr(base[sel & (yrs <= 2022)]), test=L.sr(new[sel & (yrs >= 2023)]) - L.sr(base[sel & (yrs >= 2023)]),
                             组合ΔSR=L.sr(xp[xp.index >= "2019-01-01"]) - L.sr(base_p[base_p.index >= "2019-01-01"]), p单=float(np.mean(nl2[key] >= o))))
    fam2 = np.max(np.vstack([nl2[k] for k in nl2]), axis=0)
    L2 = pd.DataFrame(rows); L2[f"p族内({len(nl2)})"] = [float(np.mean(fam2 >= ob2[(a, q)])) for a, q in zip(L2.特征, L2.q)]
    print("\n变现(跳过因果排名最低 q 的日子):"); print(L2.to_string(index=False)); L2.to_csv(TAB / "liq_signal.csv", index=False)
    print(f"\ndev/test 同号为正:{int(((L2.dev>0)&(L2.test>0)).sum())}/{len(L2)};族内 p 最小 {L2.iloc[:,-1].min():.3f}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
