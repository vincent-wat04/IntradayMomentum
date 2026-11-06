"""多源新闻语料(HF bucket 缓存)→ 决策日 D 的新闻特征。见 docs/PROPOSAL_news_multisource.md。
PIT:新闻归属第一个满足『D 的 09:30 ET ≥ 新闻时间』的交易日;日级子集按语料的 date_trading。"""
from __future__ import annotations

import os
import re
from pathlib import Path

import warnings

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(os.environ["ETFT_DATA"]) / "raw" / "news_ms" / "data"
SRC = {"sp500": "sp500_daily_headlines", "finsen": "finsen_us_2007_2023", "reddit": "reddit_finance_sp500", "yahoo": "yahoo_finance_felixdrinkall"}
COVER_END = pd.Timestamp("2023-12-29")           # 之后覆盖不足,特征置 NaN
CATS = {
    "FED": r"\b(fed|fomc|powell|federal reserve|rate hikes?|rate cuts?|interest rates?|central bank)\b",
    "INFL": r"\b(cpi|inflation|ppi|pce|consumer prices?|price index)\b",
    "JOBS": r"\b(jobs report|payrolls?|unemployment|jobless|labor market|employment)\b",
    "GEO": r"\b(war|invasion|invades?|missiles?|sanctions?|attacks?|ukraine|russia|taiwan|iran|israel|hamas|tariffs?|trade war|conflict)\b",
    "BANK": r"\b(bank failures?|svb|silicon valley bank|credit suisse|default|bankruptcy|bailout|credit crunch|contagion)\b",
    "CRASH": r"\b(crash\w*|plunge[sd]?|plummet\w*|tumble[sd]?|sell-?off|slump\w*|rout|panic|fear)\b",
    "RALLY": r"\b(surge[sd]?|soar\w*|rall(?:y|ies|ied)|record highs?|jump\w*|boom|rebound\w*)\b",
    "EARN": r"\b(earnings|eps|guidance|revenue|quarterly results?|beats?|misses)\b",
    "TECH": r"\b(nasdaq|tech|technology|ai|artificial intelligence|nvidia|apple|microsoft|semiconductors?|chips?)\b",
}
POS = r"\b(gain\w*|rise[sn]?|rising|rall\w+|surge\w*|beat\w*|strong\w*|growth|record|bull\w*|optimis\w+|rebound\w*|upgrade\w*|rall(?:y|ies)|soar\w*|boost\w*|recover\w*|advance\w*|upbeat)\b"
NEG = r"\b(loss\w*|fall\w*|fell|drop\w*|plunge\w*|miss\w*|weak\w*|recession|fear\w*|bear\w*|downgrade\w*|cuts?|crisis|slump\w*|warn\w*|tumble\w*|crash\w*|slide\w*|sink\w*|worr\w+|selloff|sell-off)\b"
_CAT_RX = {k: re.compile(v, re.I) for k, v in CATS.items()}
_POS, _NEG = re.compile(POS, re.I), re.compile(NEG, re.I)
warnings.filterwarnings("ignore")
START, END = "2018-11-01", "2024-01-03"


def load_items() -> pd.DataFrame:
    con = duckdb.connect(); out = []
    for key, ds in SRC.items():
        g = str(ROOT / ds / "*.parquet")
        q = (f"select '{key}' src, date, split_part(text, chr(10), 1) title, json_extract_string(extra_fields,'$.date_trading') dt, "
             f"json_extract_string(extra_fields,'$.time_precision') prec from read_parquet('{g}') where date >= '{START}' and date < '{END}'")
        out.append(con.execute(q).df())
    d = pd.concat(out, ignore_index=True)
    d["ts"] = pd.to_datetime(d["date"], utc=True); d["title"] = d["title"].fillna("").str.slice(0, 300)
    d["dt"] = pd.to_datetime(d["dt"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    return d.drop(columns="date")


def assign_day(items: pd.DataFrame, days: pd.DatetimeIndex, leak: bool = False) -> pd.Series:
    """返回每条新闻的决策日(NaT=丢弃)。leak=True 为阳性对照:用 16:00 收盘作截止(盘中新闻并入当日)。"""
    hh = (16, 0) if leak else (9, 30)
    cut = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {hh[0]:02d}:{hh[1]:02d}", tz="America/New_York") for d in days]).tz_convert("UTC")
    res = pd.Series(pd.NaT, index=items.index, dtype="datetime64[ns]")
    dayl = items.prec.eq("day").to_numpy()
    ts = items.ts.dt.tz_convert("UTC")
    m = ~dayl
    idx = np.searchsorted(cut.asi8, ts[m].astype("datetime64[ns, UTC]").astype("int64").to_numpy(), side="left")
    ok = idx < len(days); r = pd.Series(pd.NaT, index=items.index[m], dtype="datetime64[ns]"); r[ok] = days[idx[ok]]; res[m] = r
    dd = items.dt[dayl]
    idx2 = np.searchsorted(days.values, dd.values, side="left"); ok2 = (idx2 < len(days)) & dd.notna().to_numpy()
    r2 = pd.Series(pd.NaT, index=dd.index, dtype="datetime64[ns]"); r2[ok2] = days[idx2[ok2]]; res[dayl] = r2
    return res


USE_SRC = ("sp500", "finsen", "reddit")           # yahoo_felix 在 2021-01 覆盖从 ~2k/年跳到 ~20k/年(结构性断点),且是公司级新闻 → 不用(见 PROPOSAL §6 偏离 1)


def daily_features(items: pd.DataFrame, days: pd.DatetimeIndex, leak: bool = False) -> pd.DataFrame:
    it = items[items.src.isin(USE_SRC)].copy(); it["day"] = assign_day(it, days, leak); it = it[it.day.notna()]
    t = it.title
    for k, rx in _CAT_RX.items(): it[k] = t.str.contains(rx).to_numpy()
    it["pos"], it["neg"] = t.str.count(_POS).to_numpy(), t.str.count(_NEG).to_numpy()
    mk = it[it.src.isin(["sp500", "finsen"])]; rd = it[it.src.eq("reddit")]
    F = pd.DataFrame(index=days)
    F["n_mk"] = mk.groupby("day").size().reindex(days).fillna(0); F["n_rd"] = rd.groupby("day").size().reindex(days).fillna(0)
    for k in CATS: F["mk_" + k] = mk.groupby("day")[k].sum().reindex(days).fillna(0)
    for k in ("CRASH", "RALLY"):                                       # reddit:占当日帖子的比例(覆盖随年份增长,用占比)
        F["rd_" + k] = (rd.groupby("day")[k].sum() / rd.groupby("day").size()).reindex(days).fillna(0)
    for nm, g in (("neg_mk", mk), ("neg_rd", rd)):
        p = g.groupby("day").pos.sum().reindex(days).fillna(0); n = g.groupby("day").neg.sum().reindex(days).fillna(0)
        F[nm] = (n / (p + n)).where((p + n) >= 3)
    R = pd.DataFrame(index=days)
    for c in ["n_mk", "n_rd"] + ["mk_" + k for k in CATS]:
        x = np.log1p(F[c]); R["w_" + c] = x - x.rolling(60, min_periods=30).mean().shift(1)
    for c in ("rd_CRASH", "rd_RALLY", "neg_mk", "neg_rd"):
        mu = F[c].rolling(60, min_periods=20).mean().shift(1); R["w_" + c] = (F[c].fillna(mu) - mu).fillna(0.0)      # 词数不足/均值不可得 → 0(无信息)
    R.loc[R.index > COVER_END] = np.nan
    return R
