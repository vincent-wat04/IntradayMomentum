"""「排除周一」的离线对照组 —— 取代原来那个常驻影子进程。

为什么改成离线：原设计跑第二个常驻进程去记录周一的决策，在一台只剩约 228MB
可用内存（svt-serve 独占 2.4GB）的机器上又吃 90MB，直接把 sshd 压到 fork 不出来。
而这个进程是**完全多余的**：策略对给定的 bar 序列是确定性的，
所以只要有本地 1 分钟缓存，就能离线重放出"周一本来会做什么"。

用法：
  python live/monday_control.py --pull     # 先从服务器拉最新的 bar 缓存
  python live/monday_control.py            # 重放并对比「跑周一」vs「不跑周一」

输出：
  1. 每个周一的决策与模拟盈亏（和实盘日志同格式，便于人工核对）
  2. 累计对比：排除周一 vs 全周都跑，以及周一本身的累计贡献
  3. 随着样本外天数增加，这个对比会逐步给出 PART11 §2.3 想要的答案
     （当时 dev 置换 p=0.41、test p=0.077，证据不足以下结论）
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from strategy_signal import Cfg, DayPlan, decide, sigma_profile, daily_sigma, exposure_from

CACHE = HERE / "state" / "bars_US_QQQ.jsonl"
PER_SHARE = 0.005 + 0.0045
TRADE_BETA = 3.0          # 走 TQQQ


def pull():
    from sshx import conn, REMOTE
    c = conn(tries=15)
    s = c.open_sftp()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    s.get(f"{REMOTE}/state/bars_US_QQQ.jsonl", str(CACHE))
    s.close(); c.close()
    print(f"已拉取 {CACHE}（{CACHE.stat().st_size/1e6:.2f} MB）")


def read_cache():
    days = {}
    for line in CACHE.read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        days.setdefault(r["t"][:10], {})[r["t"][11:16]] = r
    return days


def replay_day(day, cache, cfg, tqqq_px=None):
    """离线重放单个交易日，返回 (日收益, 决策明细)。严格只用该日之前的数据算 σ/敞口。"""
    past = [d for d in sorted(cache) if d < day]
    if len(past) < max(5, int(0.7 * cfg.lookback)):
        return None, []
    row = cache[day]
    clocks = sorted(row)
    hist = []
    for d in past[-cfg.lookback:]:
        r = cache[d]
        first = min(r)
        hist.append((d, {k: v["c"] for k, v in r.items()}, r[first]["o"]))
    sig = sigma_profile(hist, clocks, cfg.lookback)
    closes = [cache[d][max(cache[d])]["c"] for d in past[-(cfg.lookback + 2):]]
    sd = daily_sigma(closes, cfg.lookback)
    e = exposure_from(sd, cfg)
    if e is None:
        return None, []
    day_open = row[min(row)]["o"]
    prev_close = cache[past[-1]][max(cache[past[-1]])]["c"]
    plan = DayPlan(day=day, day_open=day_open, prev_close_adj=prev_close,
                   sigma=sig, exposure=min(e, 4.0), sigma_daily=sd)

    # 逐 bar 重放：决策在 bar 收盘、成交在下一根 bar 开盘、收盘 MOC 平
    pos, entry, pnl, cost, logs = 0.0, np.nan, 0.0, 0.0, []
    num = den = 0.0
    px_t = tqqq_px if tqqq_px and tqqq_px > 0 else np.nan
    rate = (PER_SHARE / px_t) / TRADE_BETA if np.isfinite(px_t) else PER_SHARE / day_open
    for i, clk in enumerate(clocks[:-1]):
        bar = row[clk]
        num += (bar["h"] + bar["l"] + bar["c"]) / 3.0 * bar["v"]; den += bar["v"]
        vwap = num / den if den > 0 else float("nan")
        nxt = row[clocks[i + 1]]
        fill = nxt["o"]
        tgt, why = decide(plan, clk, bar["c"], vwap, pos, cfg,
                          entry_px=entry if pos != 0 else float("nan"))
        if tgt != pos:
            if pos != 0 and np.isfinite(entry):
                pnl += pos * (fill / entry - 1.0)
            cost += rate * abs(tgt - pos)
            logs.append(dict(clock=clk, close=bar["c"], vwap=round(vwap, 4),
                             pos=pos, target=tgt, fill=fill, why=why))
            pos = tgt
            entry = fill if tgt != 0 else np.nan
    if pos != 0 and np.isfinite(entry):
        dc = row[clocks[-1]]["c"]
        pnl += pos * (dc / entry - 1.0)
        cost += rate * abs(pos)
        logs.append(dict(clock=clocks[-1], close=dc, pos=pos, target=0.0, fill=dc,
                         why="MOC 收盘平仓"))
    return pnl - cost, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--since", default=None, help="只看这个日期之后（默认全部缓存）")
    ap.add_argument("--verbose", action="store_true", help="打印每个周一的逐笔决策")
    a = ap.parse_args()
    if a.pull:
        pull(); return

    if not CACHE.exists():
        print("没有本地缓存，先跑 --pull"); return 1
    cache = read_cache()
    cfg = Cfg()
    lev = Path("data/lev/levetf_close.csv")
    tq = None
    if lev.exists():
        d = pd.read_csv(lev, index_col=0)
        tq = d["TQQQ"] if "TQQQ" in d.columns else None

    days = sorted(cache)
    if a.since:
        days = [d for d in days if d >= a.since]
    rows = []
    for d in days:
        ts = pd.Timestamp(d)
        px = float(tq.get(d)) if (tq is not None and d in tq.index) else None
        r, logs = replay_day(d, cache, cfg, tqqq_px=px)
        if r is None:
            continue
        rows.append(dict(day=d, weekday=ts.dayofweek, ret=r, n_trades=len(logs)))
        if a.verbose and ts.dayofweek == 0 and logs:
            print(f"\n--- {d}（周一）日收益 {r*1e4:+.1f}bp")
            for g in logs:
                print(f"    {g['clock']} close={g['close']:.2f} pos={g['pos']:+.3f}"
                      f" -> {g['target']:+.3f}  {g['why'][:70]}")
    R = pd.DataFrame(rows)
    if not len(R):
        print("缓存里没有足够的历史来重放"); return 1

    mon, oth = R[R.weekday == 0], R[R.weekday != 0]
    print("=" * 92)
    print(f"离线重放 {len(R)} 个交易日（{R.day.iloc[0]} .. {R.day.iloc[-1]}），"
          f"其中周一 {len(mon)} 天")
    print("=" * 92)

    def stat(x, nm):
        x = pd.Series(x).dropna()
        if len(x) < 2:
            return f"  {nm:22s} 样本太少（{len(x)} 天）"
        s = (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")
        return (f"  {nm:22s} {len(x):4d} 天  日均 {x.mean()*1e4:+7.2f}bp  "
                f"累计 {x.sum()*100:+7.2f}%  Sharpe {s:+6.2f}  "
                f"胜率 {(x>0).mean():5.1%}")

    print(stat(R.ret, "全周都跑"))
    print(stat(oth.ret, "排除周一（实盘在跑）"))
    print(stat(mon.ret, "周一单独（对照组）"))
    print()
    if len(mon) >= 2:
        print(f"  周一的累计贡献 {mon.ret.sum()*100:+.2f}%  "
              f"-> 排除周一{'对' if mon.ret.sum() < 0 else '错'}了"
              f"（截至目前，样本 {len(mon)} 天，远不足以下结论）")
    print(f"\n  参考：PART11 §2.3 的回测结论是 test 期周一日均 −3.57bp、"
          f"置换 p=0.077（dev p=0.41）")
    print(f"  需要大约 {max(0, 60-len(mon))} 个以上的样本外周一才开始有分辨力")

    out = HERE / "state" / "monday_control.csv"
    R.to_csv(out, index=False)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
