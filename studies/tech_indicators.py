"""32 个经典技术指标(东北证券《量化择时系列一》)—— numpy/pandas 手写(TA-Lib 不可用)。
全部**因果**:第 t 个信号只用 ≤t 的 bar。输入为跨日连续的一维 bar 序列;newday[t]=True 表示该 bar 是当日第一根
(仅用于状态机在开盘重置为空仓,由日历结构决定,与价格无关)。
输出 s∈{+1,-1,0}(int8)。规则见 docs/PROPOSAL_intraday_tech_indicators.md §2。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TREND = ["SMA", "EMA", "KAMA", "MACD", "AROON", "ADX", "DPO", "SAR"]
MOMENTUM = ["MOM", "BIAS", "RSI", "ROC", "KDJ", "WR", "CCI", "CMO", "UO", "TRIX", "POS"]
VOLAT = ["KC", "BBANDS", "DC", "ACCBANDS", "MASSI", "RVI", "UDVD"]
VOLUME = ["AD", "OBV", "MFI", "EOM", "MAAMT", "FI"]
NAMES = TREND + MOMENTUM + VOLAT + VOLUME
CLASS = {**{k: "趋势" for k in TREND}, **{k: "动量" for k in MOMENTUM}, **{k: "波动" for k in VOLAT}, **{k: "成交量" for k in VOLUME}}


def _S(x): return pd.Series(x)
def sma(x, n): return _S(x).rolling(n).mean()
def ema(x, n): return _S(x).ewm(span=n, adjust=False).mean()
def wilder(x, n): return _S(x).ewm(alpha=1.0 / n, adjust=False).mean()
def _sgn(x): return np.sign(np.nan_to_num(np.asarray(x, dtype=float))).astype(np.int8)


def _state_machine(c, enter_l, exit_l, enter_s, exit_s, newday):
    """逐 bar 状态机:开盘重置为 0;多头在 exit_l 退出,空头在 exit_s 退出;入场优先于同 bar 的退出。"""
    n = len(c); s = np.zeros(n, np.int8); st = 0
    for t in range(n):
        if newday[t]: st = 0
        if st == 1 and exit_l[t]: st = 0
        elif st == -1 and exit_s[t]: st = 0
        if st == 0:
            if enter_l[t]: st = 1
            elif enter_s[t]: st = -1
        s[t] = st
    return s


def _kama(c, n=10, f=2, sl=30):
    c = np.asarray(c, float); ch = np.abs(_S(c).diff(n).to_numpy()); vol = _S(np.abs(np.diff(c, prepend=np.nan))).rolling(n).sum().to_numpy()
    er = np.where(vol > 0, ch / vol, 0.0); sc = (er * (2.0 / (f + 1) - 2.0 / (sl + 1)) + 2.0 / (sl + 1)) ** 2
    k = np.full(len(c), np.nan); st = n
    k[st] = c[st]
    for t in range(st + 1, len(c)):
        k[t] = k[t - 1] + (0.0 if np.isnan(sc[t]) else sc[t]) * (c[t] - k[t - 1])
    return k


def _sar(h, l, af0=0.02, step=0.02, mx=0.2):
    n = len(h); sar = np.full(n, np.nan); up = True; af = af0; ep = h[0]; sar[0] = l[0]
    for t in range(1, n):
        s_ = sar[t - 1] + af * (ep - sar[t - 1])
        if up:
            s_ = min(s_, l[t - 1], l[t - 2] if t >= 2 else l[t - 1])
            if l[t] < s_: up = False; s_ = ep; ep = l[t]; af = af0
            elif h[t] > ep: ep = h[t]; af = min(af + step, mx)
        else:
            s_ = max(s_, h[t - 1], h[t - 2] if t >= 2 else h[t - 1])
            if h[t] > s_: up = True; s_ = ep; ep = h[t]; af = af0
            elif l[t] < ep: ep = l[t]; af = min(af + step, mx)
        sar[t] = s_
    return sar


def macd(c, fast=12, slow=26, sig=9):
    dif = ema(c, fast) - ema(c, slow); dea = dif.ewm(span=sig, adjust=False).mean()
    return dif.to_numpy(), dea.to_numpy()


def compute_signals(o, h, l, c, v, newday, bars_per_day, only=None) -> dict:
    o, h, l, c, v = (np.asarray(x, float) for x in (o, h, l, c, v)); newday = np.asarray(newday, bool)
    C, H, L, V = _S(c), _S(h), _S(l), _S(v)
    pc = C.shift(1); out = {}
    want = lambda k: only is None or k in only
    amt = V * C
    # ---------------- 趋势
    if want("SMA"): out["SMA"] = _sgn(C - sma(c, 20))
    if want("EMA"): out["EMA"] = _sgn(C - ema(c, 20))
    if want("KAMA"): out["KAMA"] = _sgn(c - _kama(c))
    if want("MACD"):
        dif, dea = macd(c); out["MACD"] = _sgn(dif - dea)
    if want("AROON"):
        n = 25
        am = lambda a: np.argmax(a) ; an = lambda a: np.argmin(a)
        up = 100.0 * (n - (n - H.rolling(n + 1).apply(am, raw=True))) / n   # n - 距今根数
        dn = 100.0 * (n - (n - L.rolling(n + 1).apply(an, raw=True))) / n
        out["AROON"] = _sgn(up - dn)
    if want("ADX"):
        n = 14; up_m = H.diff(); dn_m = -L.diff()
        pdm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0); mdm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
        tr = pd.concat([H - L, (H - pc).abs(), (L - pc).abs()], axis=1).max(axis=1)
        atr = wilder(tr, n); pdi = 100 * wilder(pdm, n) / atr; mdi = 100 * wilder(mdm, n) / atr
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi); adx = wilder(dx.fillna(0), n)
        s = np.where((pdi > mdi) & (adx > 25), 1, np.where((mdi > pdi) & (adx > 25), -1, 0)); out["ADX"] = s.astype(np.int8)
    if want("DPO"): out["DPO"] = _sgn(C - sma(c, 20).shift(11))
    if want("SAR"): out["SAR"] = _sgn(c - _sar(h, l))
    # ---------------- 动量
    if want("MOM"): out["MOM"] = _sgn(C - C.shift(10))
    if want("BIAS"):
        b = (C - sma(c, 20)) / sma(c, 20); th = 0.01 / np.sqrt(bars_per_day)
        out["BIAS"] = np.where(b > th, 1, np.where(b < -th, -1, 0)).astype(np.int8)
    d = C.diff()
    if want("RSI"):
        g = wilder(d.clip(lower=0).fillna(0), 14); ls = wilder((-d).clip(lower=0).fillna(0), 14)
        rsi = 100 * g / (g + ls); out["RSI"] = _sgn(rsi - 50)
    if want("ROC"): out["ROC"] = _sgn(C / C.shift(12) - 1)
    if want("KDJ"):
        ll, hh = L.rolling(9).min(), H.rolling(9).max(); rsv = 100 * (C - ll) / (hh - ll)
        K = rsv.ewm(alpha=1 / 3, adjust=False).mean(); D = K.ewm(alpha=1 / 3, adjust=False).mean(); out["KDJ"] = _sgn(K - D)
    if want("WR"):
        hh, ll = H.rolling(14).max(), L.rolling(14).min(); wr = -100 * (hh - C) / (hh - ll)
        out["WR"] = np.where(wr.isna(), 0, np.where(wr > -50, 1, -1)).astype(np.int8)
    if want("CCI"):
        tp = (H + L + C) / 3; m = tp.rolling(14).mean(); mad = tp.rolling(14).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
        cci = (tp - m) / (0.015 * mad); out["CCI"] = np.where(cci > 100, 1, np.where(cci < -100, -1, 0)).astype(np.int8)
    if want("CMO"):
        su = d.clip(lower=0).rolling(14).sum(); sd_ = (-d).clip(lower=0).rolling(14).sum(); out["CMO"] = _sgn((su - sd_) / (su + sd_))
    if want("UO"):
        bp = C - np.minimum(L, pc); tr = np.maximum(H, pc) - np.minimum(L, pc)
        a = lambda n: bp.rolling(n).sum() / tr.rolling(n).sum()
        uo = 100 * (4 * a(7) + 2 * a(14) + a(28)) / 7
        out["UO"] = np.where(uo.isna(), 0, np.where(uo > 50, 1, -1)).astype(np.int8)
    if want("TRIX"):
        t3 = ema(ema(ema(c, 12), 12), 12); out["TRIX"] = _sgn(t3.pct_change())
    if want("POS"):
        ll, hh = L.rolling(20).min(), H.rolling(20).max(); pos = (C - ll) / (hh - ll)
        out["POS"] = np.where(pos.isna(), 0, np.where(pos > 0.5, 1, -1)).astype(np.int8)
    # ---------------- 波动(通道突破,开盘重置)
    def channel(upper, lower, mid, key):
        u, lo_, m = np.asarray(upper), np.asarray(lower), np.asarray(mid)
        u1, l1 = np.roll(u, 1), np.roll(lo_, 1); u1[0] = l1[0] = np.nan
        ok = np.isfinite(u1) & np.isfinite(l1) & np.isfinite(m)
        out[key] = _state_machine(c, ok & (c > u1), ok & (c < m), ok & (c < l1), ok & (c > m), newday)
    if want("KC"):
        tr = pd.concat([H - L, (H - pc).abs(), (L - pc).abs()], axis=1).max(axis=1); atr = tr.rolling(10).mean(); mid = ema(c, 20)
        channel((mid + 2 * atr).to_numpy(), (mid - 2 * atr).to_numpy(), mid.to_numpy(), "KC")
    if want("BBANDS"):
        m = sma(c, 20); sd_ = C.rolling(20).std(ddof=0); channel((m + 2 * sd_).to_numpy(), (m - 2 * sd_).to_numpy(), m.to_numpy(), "BBANDS")
    if want("DC"):
        # 上/下轨 = 前 20 根(不含当前)的最高/最低;中点出场
        u0 = H.rolling(20).max().to_numpy(); l0 = L.rolling(20).min().to_numpy()
        u1, l1 = np.roll(u0, 1), np.roll(l0, 1); u1[0] = l1[0] = np.nan
        mid1 = (u1 + l1) / 2
        out["DC"] = _state_machine(c, np.isfinite(u1) & (c > u1), np.isfinite(mid1) & (c < mid1), np.isfinite(l1) & (c < l1), np.isfinite(mid1) & (c > mid1), newday)
    if want("ACCBANDS"):
        f = 4 * (H - L) / (H + L); up = (H * (1 + f)).rolling(20).mean(); lo_ = (L * (1 - f)).rolling(20).mean()
        channel(up.to_numpy(), lo_.to_numpy(), sma(c, 20).to_numpy(), "ACCBANDS")
    if want("MASSI"):
        r = H - L; e1 = ema(r, 9); e2 = e1.ewm(span=9, adjust=False).mean(); mi = (e1 / e2).rolling(25).sum()
        trig = ((mi.shift(1).rolling(25).max() > 27) & (mi < 26.5) & (mi.shift(1) >= 26.5)).to_numpy()
        dirn = (-_sgn(C - ema(c, 9))).astype(np.int8); s = np.zeros(len(c), np.int8); hold = 0; cur = 0
        for t in range(len(c)):
            if newday[t]: hold = 0; cur = 0
            if trig[t]: hold = 9; cur = dirn[t]
            if hold > 0: s[t] = cur; hold -= 1
        out["MASSI"] = s
    if want("RVI"):
        sd_ = C.rolling(10).std(ddof=0); ch = C.diff(); u = sd_.where(ch > 0, 0.0).fillna(0); dd = sd_.where(ch < 0, 0.0).fillna(0)
        un, dn = ema(u, 14), ema(dd, 14); rvi = 100 * un / (un + dn)
        out["RVI"] = np.where(rvi.isna(), 0, np.where(rvi > 50, 1, -1)).astype(np.int8)
    if want("UDVD"):
        r = C.pct_change(); up2 = (r ** 2).where(r > 0, 0.0).rolling(20).mean(); dn2 = (r ** 2).where(r < 0, 0.0).rolling(20).mean()
        out["UDVD"] = _sgn(np.sqrt(up2) - np.sqrt(dn2))
    # ---------------- 成交量
    if want("AD"):
        clv = ((C - L) - (H - C)) / (H - L); clv = clv.where(H > L, 0.0).fillna(0); ad = (clv * V).cumsum(); out["AD"] = _sgn(ad - ad.rolling(20).mean())
    if want("OBV"):
        obv = (np.sign(d.fillna(0)) * V).cumsum(); out["OBV"] = _sgn(obv - obv.rolling(20).mean())
    if want("MFI"):
        tp = (H + L + C) / 3; mf = tp * V; dtp = tp.diff()
        pos_ = mf.where(dtp > 0, 0.0).rolling(14).sum(); neg_ = mf.where(dtp < 0, 0.0).rolling(14).sum(); mfi = 100 * pos_ / (pos_ + neg_)
        out["MFI"] = np.where(mfi.isna(), 0, np.where(mfi > 50, 1, -1)).astype(np.int8)
    if want("EOM"):
        mid = (H + L) / 2; dm = mid.diff(); br = (V / (H - L)).replace([np.inf, -np.inf], np.nan)
        emv = (dm / br).replace([np.inf, -np.inf], np.nan).fillna(0); out["EOM"] = _sgn(emv.rolling(14).mean())
    if want("MAAMT"):
        out["MAAMT"] = np.where(amt.rolling(5).mean() > amt.rolling(20).mean(), 1, 0).astype(np.int8)
    if want("FI"):
        out["FI"] = _sgn(ema((d * V).fillna(0), 13))
    return out
