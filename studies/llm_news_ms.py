"""第二阶段:多源新闻头条 → LLM 结构化事件抽取 → 次日日内波动检验。docs/PROPOSAL_news_multisource.md「第二阶段补充登记」。
  export SOCLAAS_API_KEY=... SOCLAAS_BASE_URL=...     # 只经环境变量,不写文件
  python3 llm_news_ms.py --run      # 抽取并缓存
  python3 llm_news_ms.py --eval     # 聚合特征 + 评估"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_event_extract as X
import lowvol_round4_news_study as S4
import news_ms as N
from etft.intraday.bars import load_bars

MODEL = "llama3.1:8b"


def build_windows(days):
    items = N.load_items(); it = items[items.src.isin(["sp500", "finsen"])].copy()
    it["day"] = N.assign_day(it, days); it = it[it.day.notna() & (it.day >= "2019-01-01") & (it.day <= N.COVER_END)]
    it = it[it.title.str.len() > 8]
    W = {}
    for d, g in it.groupby("day"):
        W[pd.Timestamp(d)] = pd.DataFrame({"text": g.title.to_numpy(), "ts": g.ts.to_numpy(), "importance": 0.0})
    return W


def features(days, W):
    dd = pd.DatetimeIndex(sorted(W))
    A = X.aggregate(MODEL, dd, W)
    cov = A.n_parsed.sum() / max(A.n_items.sum(), 1)
    print(f"解析覆盖率 {cov:.3f};有新闻的天数 {len(dd)};条目总数 {int(A.n_items.sum())}")
    F = A.reindex(days)[X.FEATS].copy()
    raw = F.copy()
    # 无新闻的天数 = 0 计数;相对化:减去此前 60 日均值(shift 1)
    R = pd.DataFrame(index=days)
    for c in X.FEATS:
        x = F[c].fillna(0.0); R["w_" + c] = x - x.rolling(60, min_periods=30).mean().shift(1)
    R.loc[R.index > N.COVER_END] = np.nan
    return R, cov


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--eval", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); days = load_bars("QQQ").days
    W = build_windows(days); dd = pd.DatetimeIndex(sorted(W))
    print(f"窗口 {len(dd)} 天;每天条目 中位 {int(np.median([len(W[d]) for d in dd]))},最大 {max(len(W[d]) for d in dd)};批次约 {sum(int(np.ceil(min(len(W[d]), X.MAXN) / X.BATCH)) for d in dd)}")
    if a.run: X.run(MODEL, dd, W, workers=a.workers)
    if a.eval:
        R, cov = features(days, W)
        if cov < 0.9: print("解析覆盖率<0.9,停止(提示词/模型输出不稳定,不得改提示词重跑后选择性报告)"); return
        S4.evaluate(R, "_llm")


if __name__ == "__main__":
    main()
