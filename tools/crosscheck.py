"""实盘状态 vs 回测状态 的对账：最后一道正确性关卡。

实盘的 σ profile / 日波动 / 敞口 是从服务器上的本地 1 分钟缓存（moomoo 实时流 + 历史灌入）
算出来的；回测用的是 etf_timing 的 min1 数据经 load_bars() 处理后的面板。两条路径
完全独立。如果它们在重叠日期上给出相同的 σ 和敞口，就说明实盘读到的状态和回测一致。

跑法：
  1. 先把服务器上的缓存拉下来：python live/crosscheck.py --pull
  2. 再对账：python live/crosscheck.py
"""
from __future__ import annotations

import _paths  # noqa: F401  (设置 sys.path)

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from etft.intraday.bars import load_bars
from live import strategy_signal as SG

LOCAL_CACHE = HERE / "state" / "bars_US_QQQ.jsonl"


def pull():
    from sshx import conn, REMOTE
    c = conn()
    s = c.open_sftp()
    LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    s.get(f"{REMOTE}/state/bars_US_QQQ.jsonl", str(LOCAL_CACHE))
    s.close(); c.close()
    print(f"已拉取 {LOCAL_CACHE}（{LOCAL_CACHE.stat().st_size/1e6:.2f} MB）")


def read_cache():
    days = {}
    for line in LOCAL_CACHE.read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        days.setdefault(r["t"][:10], {})[r["t"][11:16]] = r
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    a = ap.parse_args()
    if a.pull:
        pull(); return 0

    cache = read_cache()
    # QQQ 的除息日与每股股息（来自 moomoo get_rehab，已验证：原始收盘 − 股息 = 前复权价）
    DIVS = {"2025-09-22": 0.69395, "2025-12-22": 0.79408, "2026-03-23": 0.73282,
            "2026-06-22": 0.81349, "2026-09-21": 0.75143}
    b = load_bars("QQQ")
    cfg = SG.Cfg()
    clocks = list(b.c.columns)

    bt_days = {str(d.date()) for d in b.days}
    overlap = sorted(set(cache) & bt_days)
    print(f"实盘缓存 {len(cache)} 天（{min(cache)} .. {max(cache)}）；"
          f"回测 {len(bt_days)} 天；重叠 {len(overlap)} 天")
    if len(overlap) < 5:
        print("重叠太少，无法对账"); return 1

    # 回测侧的 σ profile / 日波动（和 replay.py 一致）
    move = (b.c.div(b.day_open, axis=0) - 1).abs()
    SIG = move.rolling(cfg.lookback, min_periods=max(5, int(0.7 * cfg.lookback))).mean().shift(1)
    rd = b.day_close / b.prev_close_adj - 1
    SD = rd.rolling(cfg.lookback, min_periods=max(5, cfg.lookback // 2)).std().shift(1)

    # ---- 1. 逐 bar 收盘价对账（最基础）
    rows = []
    for d in overlap:
        ts = pd.Timestamp(d)
        for clk in ("09:31", "10:00", "12:00", "15:00", "16:00"):
            a_ = cache[d].get(clk, {}).get("c")
            e_ = b.c.loc[ts, clk] if clk in b.c.columns else np.nan
            if a_ is not None and np.isfinite(e_):
                rows.append(dict(day=d, clock=clk, live=a_, bt=e_,
                                 rel=abs(a_ / e_ - 1)))
    P = pd.DataFrame(rows)
    print(f"\n[1] bar 收盘价对账：{len(P)} 个点，中位相对差 {P.rel.median():.2e}，"
          f"p99 {P.rel.quantile(.99):.2e}，最大 {P.rel.max():.2e}")
    ok1 = P.rel.max() < 1e-6
    print(f"    {'[PASS]' if ok1 else '[**FAIL**]'} 两条数据链路的 bar 收盘价一致")
    if not ok1:
        print(P.nlargest(5, "rel").to_string(index=False))

    # ---- 2. σ profile 对账（需要缓存里有足够的历史日）
    cdays = sorted(cache)
    rows2 = []
    for d in overlap:
        i = cdays.index(d)
        past = cdays[max(0, i - cfg.lookback):i]
        if len(past) < cfg.lookback:
            continue
        hist = []
        for p in past:
            row = cache[p]
            first = min(row)
            hist.append((p, {k: v["c"] for k, v in row.items()}, row[first]["o"]))
        sig = SG.sigma_profile(hist, clocks, cfg.lookback)
        ts = pd.Timestamp(d)
        for clk in ("10:00", "11:00", "12:00", "13:00", "14:00", "15:00"):
            a_ = sig.get(clk)
            e_ = SIG.loc[ts, clk] if clk in SIG.columns else np.nan
            if a_ is not None and np.isfinite(e_):
                rows2.append(dict(day=d, clock=clk, live=a_, bt=e_, rel=abs(a_ / e_ - 1)))
    S = pd.DataFrame(rows2)
    if len(S):
        print(f"\n[2] σ profile 对账：{len(S)} 个点（{S.day.nunique()} 天 × 6 时点），"
              f"中位相对差 {S.rel.median():.2e}，最大 {S.rel.max():.2e}")
        ok2 = S.rel.max() < 1e-6
        print(f"    {'[PASS]' if ok2 else '[**FAIL**]'} 实盘 σ 与回测 σ 一致")
        if not ok2:
            print(S.nlargest(5, "rel").to_string(index=False))
    else:
        ok2 = False
        print("\n[2] σ profile 对账：缓存历史不足（需要 lookback+1 天以上的连续历史）")

    # ---- 3. 敞口对账（日波动来自日收盘，实盘用原始收盘、回测用复权前收，会有股息差异）
    rows3 = []
    for d in overlap:
        i = cdays.index(d)
        past = cdays[max(0, i - cfg.lookback - 1):i]
        if len(past) < cfg.lookback + 1:
            continue
        closes = [cache[p][max(cache[p])]["c"] for p in past]
        dv = [DIVS.get(p, 0.0) for p in past]
        sd = SG.daily_sigma(closes, cfg.lookback, divs=dv)
        e_live = SG.exposure_from(sd, cfg)
        ts = pd.Timestamp(d)
        e_bt = SG.exposure_from(float(SD.loc[ts]) if np.isfinite(SD.loc[ts]) else None, cfg)
        if e_live and e_bt:
            rows3.append(dict(day=d, sd_live=sd, sd_bt=float(SD.loc[ts]),
                              e_live=e_live, e_bt=e_bt, rel=abs(e_live / e_bt - 1)))
    E = pd.DataFrame(rows3)
    if len(E):
        print(f"\n[3] 敞口对账：{len(E)} 天，中位相对差 {E.rel.median():.2e}，"
              f"最大 {E.rel.max():.2e}")
        ok3 = E.rel.max() < 1e-3
        print(f"    {'[PASS]' if ok3 else '[**FAIL**]'} 敞口一致（容差 1e-3，已做除息复权）")
        print(E.tail(5).round(6).to_string(index=False))
    else:
        ok3 = False
        print("\n[3] 敞口对账：缓存历史不足")

    allok = ok1 and ok2 and ok3
    print(f"\n{'='*70}\n实盘/回测状态对账：{'全部通过' if allok else '**有失败项**'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
