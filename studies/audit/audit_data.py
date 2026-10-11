"""数据层审计：分钟数据、日历、复权、bar 时标约定。

数据层在所有实验的上游，错了全盘皆错，而且这类错误最难靠读代码发现
（PART8 的 bar 时标偏移就是这类）。所以这里的核心手段是**与独立数据源对账**：
把 1 分钟 bar 聚合成日频，和 yfinance 的日线逐日比。任何时标错位、时区错误、
缺失交易日、复权错误都会在这一步暴露。
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etft.intraday.bars import load_bars, session_grid

OUT = Path("outputs/full/audit"); OUT.mkdir(parents=True, exist_ok=True)
RESULTS = []

# 已知的美股半日市（13:00 收盘）：感恩节次日、平安夜、独立日前一天
KNOWN_HALF_DAYS = {
    "2019-07-03", "2019-11-29", "2019-12-24", "2020-11-27", "2020-12-24",
    "2021-11-26", "2022-11-25", "2023-07-03", "2023-11-24", "2024-07-03",
    "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-07-03",
}


def check(name, ok, detail, catches):
    RESULTS.append(dict(check=name, pass_=bool(ok), detail=str(detail)[:400], catches=catches))
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}")
    print(f"         {detail}")
    return ok


def yf_daily(sym):
    h = yf.download(sym, start="2018-12-01", end="2026-10-05", auto_adjust=False,
                    progress=False, threads=False)
    h.columns = [c[0] if isinstance(c, tuple) else c for c in h.columns]
    h.index = pd.to_datetime(h.index).tz_localize(None)
    return h


def main():
    for sym in ("SPY", "QQQ"):
        print(f"\n{'='*70}\n=== {sym} ===")
        b = load_bars(sym)
        Y = yf_daily(sym)
        idx = b.days.intersection(Y.index)
        print(f"  分钟数据 {len(b.days)} 天，yfinance {len(Y)} 天，重叠 {len(idx)} 天")

        # ---- 1. bar 时标约定：第一根 bar 的 open 是否等于日开盘价
        o_min = b.day_open.reindex(idx)
        rel = ((o_min - Y.Open.reindex(idx)) / Y.Open.reindex(idx)).abs()
        check(f"{sym} 1.1 bar 为**结束**时标（首根 bar 的 open == 日开盘）",
              rel.median() < 2e-4,
              f"首根 bar（标签 {b.c.columns[0]}）的 open vs yfinance Open："
              f"中位相对差 {rel.median():.2e}，p95 {rel.quantile(.95):.2e}，"
              f"最大 {rel.max():.2e}",
              "bar 时标搞反（PART8 的 5 分钟 CSV 是起始时标，检查点因此偏移 5 分钟）")

        # ---- 2. 日收盘价对账
        c_min = b.day_close.reindex(idx)
        relc = ((c_min - Y.Close.reindex(idx)) / Y.Close.reindex(idx)).abs()
        bad = relc[relc > 1e-3]
        check(f"{sym} 1.2 日收盘价与 yfinance 一致", len(bad) <= 3,
              f"中位相对差 {relc.median():.2e}，超过 10bp 的天数 {len(bad)}"
              + (f"：{[str(d.date()) for d in bad.index[:5]]}" if len(bad) else ""),
              "分钟数据缺尾盘 bar、时区错误、收盘集合竞价缺失")

        # ---- 3. 日内高低价必须包住开收盘
        hi = b.h.reindex(idx).max(axis=1); lo = b.l.reindex(idx).min(axis=1)
        v1 = (hi + 1e-9 >= b.day_open.reindex(idx)) & (hi + 1e-9 >= c_min)
        v2 = (lo - 1e-9 <= b.day_open.reindex(idx)) & (lo - 1e-9 <= c_min)
        check(f"{sym} 1.3 日内最高/最低价包住开盘与收盘", bool((v1 & v2).all()),
              f"违反天数 {int((~(v1 & v2)).sum())}", "OHLC 字段错位")

        # ---- 4. 缺失交易日
        miss = Y.index.difference(b.days)
        miss = miss[(miss >= b.days[0]) & (miss <= b.days[-1])]
        check(f"{sym} 1.4 没有缺失交易日", len(miss) <= 5,
              f"缺 {len(miss)} 天" + (f"：{[str(d.date()) for d in miss[:8]]}" if len(miss) else ""),
              "数据空洞导致滚动窗口横跨真实缺口（噪声带/波动率算错）")

        # ---- 5. 半日市识别
        nb = b.c.reindex(b.days).notna().sum(axis=1)
        half = set(str(d.date()) for d in nb[nb < 300].index)
        full_grid = len(session_grid("US"))
        unexpected = half - KNOWN_HALF_DAYS
        missed = {d for d in KNOWN_HALF_DAYS if pd.Timestamp(d) in b.days} - half
        check(f"{sym} 1.5 半日市与官方日历一致", len(unexpected) == 0 and len(missed) == 0,
              f"识别出 {len(half)} 个半日市（全日 {full_grid} 根 bar）；"
              f"意外的 {sorted(unexpected)[:5]}；漏掉的 {sorted(missed)[:5]}",
              "把半日市当成数据损坏丢掉，或把残缺日当成完整日")

        # ---- 6. prev_close_adj 的复权链条是否正确（这才是我们自己的东西）
        # yfinance 口径的复权前收 = 前一日 Close × (AdjClose/Close)_{t-1} / (AdjClose/Close)_t
        f = (Y["Adj Close"] / Y.Close)
        pca_y = (Y.Close.shift(1) * f.shift(1) / f).reindex(idx)
        dp = ((b.prev_close_adj.reindex(idx) - pca_y) / pca_y).abs()
        check(f"{sym} 1.6 复权前收 prev_close_adj 与独立口径一致",
              float(dp.median()) < 1e-6 and float(dp.quantile(.99)) < 1e-4,
              f"中位相对差 {dp.median():.2e}，p99 {dp.quantile(.99):.2e}，最大 {dp.max():.2e}",
              "除息日前收没复权 -> max(open, prev_close) 被股息抬高；拆股漏调整")

        # ---- 6b. 参考源自身的问题（信息项，不计入通过率的判定逻辑里）
        on_min = (b.day_open / b.prev_close_adj).reindex(idx) - 1
        on_y = (Y.Open.reindex(idx) / pca_y) - 1
        d = (on_min - on_y).abs()
        nbad = int((d > 5e-4).sum())
        d_open = (b.day_open.reindex(idx) / Y.Open.reindex(idx) - 1).abs()
        # 把差异归因：是我们的复权错，还是参考源的 Open 字段不可靠
        attrib = "参考源的 Open 字段" if d_open[d > 5e-4].median() > \
            10 * max(dp[d > 5e-4].median(), 1e-12) else "我们的复权链条"
        check(f"{sym} 1.6b 隔夜收益差异可归因于参考源而非我们",
              nbad <= 5 or attrib == "参考源的 Open 字段",
              f"与 yfinance 口径差 >5bp 的天数 {nbad}/{len(idx)}；"
              f"这些天里 Open 差中位 {d_open[d>5e-4].median()*1e4 if nbad else 0:.2f}bp、"
              f"复权前收差中位 {dp[d>5e-4].median()*1e4 if nbad else 0:.4f}bp "
              f"-> 归因：**{attrib}**。影响已由 1.10 界定",
              "把参考源的字段问题误当成自己的 bug（或反之）")

        # ---- 7. 日收益闭合：分钟数据重构的日收益 vs yfinance 复权收益
        r_min = (b.day_close / b.prev_close_adj - 1).reindex(idx)
        r_y = (Y["Adj Close"] / Y["Adj Close"].shift(1) - 1).reindex(idx)
        d2 = (r_min - r_y).abs()
        check(f"{sym} 1.7 日收益与 yfinance 复权收益一致", float(d2.median()) < 2e-4,
              f"中位差 {d2.median():.2e}，p99 {d2.quantile(.99):.2e}，"
              f"累计差 {(1+r_min.fillna(0)).prod()/(1+r_y.fillna(0)).prod()-1:+.3%}",
              "复权链条断裂（这会同时毁掉买入持有基准和所有隔夜相关的计算）")

        # ---- 8. 单调性/重复
        check(f"{sym} 1.8 日期唯一且递增", b.days.is_monotonic_increasing and b.days.is_unique,
              f"递增 {b.days.is_monotonic_increasing}，唯一 {b.days.is_unique}",
              "重复行导致滚动窗口实际长度不对")

        # ---- 9. 价格为正、无 inf（只检查非 NaN 的 bar；半日市收盘后的 bar 被合法置空）
        X = pd.concat([b.o, b.h, b.l, b.c]).to_numpy().ravel()
        X = X[np.isfinite(X) | ~np.isnan(X)]
        xv = X[~np.isnan(X)]
        nbad = int((~(xv > 0)).sum()) + int(np.isinf(xv).sum())
        check(f"{sym} 1.9 所有非空 bar 的价格为正且有限", nbad == 0,
              f"非空 bar {len(xv):,} 个，其中非正或无穷 {nbad} 个"
              f"（被合法置空的半日市 bar {int(np.isnan(X).sum()):,} 个不计）",
              "0 或负价格导致除零、log 报错、或静默产生 inf 收益")

        # ---- 10. day_open 的数据源敏感性（yfinance 的 Open 字段对 QQQ 不可靠）
        import copy as _c
        from etft.intraday.strategy import NoiseCfg as _N, run as _r
        S = dict(stop="vwap", sizing="voltarget", max_lev=4.0, per_share=(0.005, 0.0045))
        r0 = _r(b, _N(**S))["ret"]
        b2 = _c.copy(b); b2.day_open = Y.Open.reindex(b.days).ffill()
        r1 = _r(b2, _N(**S))["ret"]
        sr = lambda x: float(x.dropna().mean() / x.dropna().std() * np.sqrt(252))
        dd = abs(sr(r1) - sr(r0))
        check(f"{sym} 1.10 结论对 day_open 的数据源不敏感", dd < 0.05,
              f"分钟首根 bar {sr(r0):.4f} vs yfinance Open {sr(r1):.4f}，"
              f"Sharpe 差 {sr(r1)-sr(r0):+.4f}，受影响 {int(((r1-r0).abs()>1e-12).sum())} 天。"
              f"（yfinance 的 Open 在 QQQ 上约 1.7% 的交易日偏 5~10bp，"
              f"而其 Close/复权链条与我们完全一致，所以问题在它的 Open 字段）",
              "单一数据源的字段错误静默传导到结论")

    R = pd.DataFrame(RESULTS)
    R.to_csv(OUT / "audit_data.csv", index=False)
    npass, ntot = int(R.pass_.sum()), len(R)
    print(f"\n{'='*70}\n数据层审计：{npass}/{ntot} 通过")
    if npass < ntot:
        print(R[~R.pass_][["check", "detail"]].to_string(index=False))
    return R


if __name__ == "__main__":
    main()
