"""LLM(新闻 ± 宏观上下文)预测 10:00 之后波动 —— 严格按 docs/PROPOSAL_llm_news_vol.md 执行。

环境变量:SOCLAAS_API_KEY、SOCLAAS_BASE_URL(不写入任何文件)。
用法(studies/ 目录):
  python llm_news_vol_study.py --prepare            # 只构建窗口/匿名化/宏观块并打印样例(不调用 API)
  python llm_news_vol_study.py --pilot              # 4 天 × 模型 × 臂,打印原始输出与 token 用量
  python llm_news_vol_study.py --probe  MODEL       # 日期探针(60 个匿名窗口)
  python llm_news_vol_study.py --run MODEL[,MODEL]  # 全量(缓存可续跑)
  python llm_news_vol_study.py --eval MODEL[,MODEL] # 评估(只读缓存)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _P
import lowvol_avoidance_study as LV

OUT = LV.OUT; TAB = LV.TAB
CACHE = OUT / "llm_cache.jsonl"
NEWS = _P.RAW.parent / "news"
START = pd.Timestamp("2026-04-02")
MAX_ITEMS, MAX_CHARS = 60, 9000
ET = "America/New_York"
ARMS = ("N", "N+M", "M")

PROMPT_VERSION = "v1"
SYSTEM = ("You are a careful market volatility analyst. You are given anonymized information available at a fixed moment: "
          "30 minutes after the U.S. stock market opened. All dates and absolute index levels have been removed. "
          "Answer with a single JSON object and nothing else.")
TASK = ("Task: judge the realized volatility of the NASDAQ-100 ETF (QQQ) over the REST of today's regular session "
        "(from 30 minutes after the open until the close), relative to a typical trading day.\n"
        'Return JSON only: {"vol_score": <integer 1-10, 1=unusually quiet, 5=typical, 10=extremely volatile>, '
        '"p_quiet": <integer 0-100, probability that the net price move over the rest of the session is among the smallest 40% of typical days>}')


# ------------------------------------------------------------------ 匿名化
_RE = [
    (re.compile(r"20\d{2}\s*年"), ""), (re.compile(r"\b20\d{2}\b"), ""),
    (re.compile(r"\d{1,2}\s*月\s*\d{1,2}\s*日"), ""), (re.compile(r"\d{1,2}\s*月份?"), ""),
    (re.compile(r"(?<!\d)\d{1,2}\s*日(?![内以])"), ""), (re.compile(r"(周|星期|礼拜)[一二三四五六日天]"), ""),
    (re.compile(r"\d{1,2}\s*[:：]\s*\d{2}"), ""), (re.compile(r"(当地|北京|美东|美国东部)?时间"), ""),
    (re.compile(r"(华尔街见闻|财联社|新华社|路透社?|彭博社?)(?=[，,：:]|\s|讯|报道|消息)"), "媒体"),
    (re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
                r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{0,2},?\s*"), ""),
    (re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri)\b\.?,?"), ""),
    (re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*点"), "[LEVEL]点"), (re.compile(r"(?<![\d.])\d{4,6}(?:\.\d+)?\s*点"), "[LEVEL]点"),
    (re.compile(r"(?<![\d.])\d{4,6}(?:\.\d+)?(?=\s*(美元|元|附近|上方|下方|关口))"), "[LEVEL]"),
    (re.compile(r"\$\s?\d{4,}(?:,\d{3})*(?:\.\d+)?"), "$[LEVEL]"),
]


def anonymize(t: str) -> str:
    for rx, rep in _RE:
        t = rx.sub(rep, t)
    return re.sub(r"\s+", " ", t).strip(" ，,;；")


def load_news() -> pd.DataFrame:
    a = pd.read_json(NEWS / "archive_us-stock-channel.jsonl", lines=True)
    b = pd.read_json(NEWS / "items.jsonl", lines=True)
    d = pd.concat([a, b]).drop_duplicates("uid")
    d["ts"] = pd.to_datetime(d.ts_utc, utc=True)
    d = d[d.text.str.len() > 8].sort_values("ts").reset_index(drop=True)
    return d


def windows(days, news):
    """每个交易日:[上一交易日 16:00 ET, 当日 10:00 ET) 的新闻。返回 {day: DataFrame}。"""
    ts_et = news.ts.dt.tz_convert(ET)
    out = {}
    for i, d in enumerate(days):
        if d < START or i == 0:
            continue
        lo = pd.Timestamp(days[i - 1].strftime("%Y-%m-%d") + " 16:00", tz=ET)
        hi = pd.Timestamp(d.strftime("%Y-%m-%d") + " 10:00", tz=ET)
        m = (ts_et >= lo) & (ts_et < hi)
        out[d] = news[m.to_numpy()]
    return out


def news_block(w: pd.DataFrame) -> str:
    if not len(w):
        return "(no news items in this window)"
    w = w.assign(n=w.importance.fillna(0)).sort_values(["n", "ts"], ascending=[False, True]).head(MAX_ITEMS).sort_values("ts")
    lines, tot = [], 0
    for t in w.text:
        s = "- " + anonymize(t)
        if tot + len(s) > MAX_CHARS:
            break
        lines.append(s); tot += len(s)
    return "NEWS (overnight and pre-market through 30 minutes after the open; anonymized):\n" + "\n".join(lines)


# ------------------------------------------------------------------ 宏观上下文(t−1 收盘,因果分位)
def macro_frame(days):
    import yfinance as yf
    tick = ["JPY=X", "AUDJPY=X", "^TNX", "^IRX", "DX-Y.NYB", "HYG", "LQD", "CL=F", "^N225", "QQQ"]
    y = yf.download(tick, start="2024-06-01", end="2026-10-12", auto_adjust=True, progress=False, threads=False)["Close"]
    y.index = pd.to_datetime(y.index).tz_localize(None)
    cb = {n: LV.cboe_prev(n, days) for n in ("VIX", "VIX9D", "VIX3M", "VXN", "SKEW")}
    yd = y.reindex(days.union(y.index)).ffill().reindex(days).shift(1)                 # t−1 收盘
    def pct(s, w=252):                                                                # 因果分位:当前值在过去 w 日中的位置(含当前,s 已是 t−1)
        return s.rolling(w, min_periods=120).apply(lambda a: (a[:-1] < a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
    X = pd.DataFrame(index=days)
    X["vix"] = cb["VIX"]; X["vix_pct"] = pct(cb["VIX"]); X["vxn"] = cb["VXN"]
    X["vix9d_vix"] = cb["VIX9D"] / cb["VIX"]; X["vix3m_vix"] = cb["VIX3M"] / cb["VIX"]; X["skew"] = cb["SKEW"]; X["skew_pct"] = pct(cb["SKEW"])
    q = yd["QQQ"]
    X["qqq_1d"] = q.pct_change(1); X["qqq_5d"] = q.pct_change(5); X["qqq_20d"] = q.pct_change(20); X["qqq_vs200"] = q / q.rolling(200).mean() - 1
    X["us10y_pct"] = pct(yd["^TNX"]); X["us10y_5d_bp"] = yd["^TNX"].diff(5) * 100
    X["curve_pct"] = pct(yd["^TNX"] - yd["^IRX"])
    X["dxy_5d"] = yd["DX-Y.NYB"].pct_change(5); X["dxy_pct"] = pct(yd["DX-Y.NYB"])
    X["usdjpy_pct"] = pct(yd["JPY=X"]); X["usdjpy_5d"] = yd["JPY=X"].pct_change(5); X["usdjpy_20d"] = yd["JPY=X"].pct_change(20)
    X["audjpy_20d"] = yd["AUDJPY=X"].pct_change(20)
    X["credit_20d"] = (yd["HYG"] / yd["LQD"]).pct_change(20)
    X["oil_5d"] = yd["CL=F"].pct_change(5); X["n225_5d"] = yd["^N225"].pct_change(5)
    X["gex_z"] = pd.read_pickle(OUT / "lowvol_study.pkl")["X"].gex_z.reindex(days)
    return X


def macro_block(r: pd.Series) -> str:
    f = lambda x, p=1: "n/a" if pd.isna(x) else f"{x:.{p}f}"
    pc = lambda x: "n/a" if pd.isna(x) else f"{100*x:+.1f}%"
    pp = lambda x: "n/a" if pd.isna(x) else f"{100*x:.0f}th pct (trailing 1y)"
    return ("MACRO CONTEXT (as of the previous U.S. close; no dates):\n"
            f"- VIX {f(r.vix)} ({pp(r.vix_pct)}); VXN {f(r.vxn)}; VIX9D/VIX {f(r.vix9d_vix,2)}; VIX3M/VIX {f(r.vix3m_vix,2)}; SKEW {f(r['skew'],0)} ({pp(r.skew_pct)})\n"
            f"- QQQ: 1d {pc(r.qqq_1d)}, 5d {pc(r.qqq_5d)}, 20d {pc(r.qqq_20d)}, vs 200d avg {pc(r.qqq_vs200)}\n"
            f"- US 10y yield level {pp(r.us10y_pct)}, 5d change {f(r.us10y_5d_bp,0)}bp; 10y-3m curve {pp(r.curve_pct)}\n"
            f"- Dollar index 5d {pc(r.dxy_5d)} ({pp(r.dxy_pct)}); USDJPY {pp(r.usdjpy_pct)}, 5d {pc(r.usdjpy_5d)}, 20d {pc(r.usdjpy_20d)}; AUDJPY 20d {pc(r.audjpy_20d)} (carry-trade proxies)\n"
            f"- Credit (HYG/LQD) 20d {pc(r.credit_20d)}; crude oil 5d {pc(r.oil_5d)}; Nikkei 5d {pc(r.n225_5d)}\n"
            f"- Dealer gamma exposure (SPX, z-score vs 1y): {f(r.gex_z,2)}")


def make_prompt(arm, w, mrow):
    parts = []
    if "N" in arm.split("+"):
        parts.append(news_block(w))
    if "M" in arm.split("+"):
        parts.append(macro_block(mrow))
    return "\n\n".join(parts) + "\n\n" + TASK


# ------------------------------------------------------------------ LLM 调用(缓存、重试、并发)
_lock = threading.Lock()


def _key(model, arm, day, prompt):
    return hashlib.sha1(f"{PROMPT_VERSION}|{model}|{arm}|{day}|{prompt}|{SYSTEM}".encode()).hexdigest()


def load_cache():
    c = {}
    if CACHE.exists():
        for ln in CACHE.read_text().splitlines():
            try:
                r = json.loads(ln); c[r["key"]] = r
            except Exception:
                pass
    return c


def client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["SOCLAAS_API_KEY"], base_url=os.environ["SOCLAAS_BASE_URL"].rstrip("/"), timeout=180, max_retries=0)


def parse(txt):
    m = re.search(r"\{.*?\}", txt, re.S)
    try:
        j = json.loads(m.group(0)); return float(j["vol_score"]), float(j["p_quiet"])
    except Exception:
        a = re.search(r'vol_score\D+(\d+(?:\.\d+)?)', txt); b = re.search(r'p_quiet\D+(\d+(?:\.\d+)?)', txt)
        return (float(a.group(1)) if a else np.nan, float(b.group(1)) if b else np.nan)


def call(cl, model, system, user, max_tokens=300):
    for att in range(5):
        try:
            r = cl.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                                           messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return r.choices[0].message.content or "", (r.usage.prompt_tokens, r.usage.completion_tokens) if r.usage else (0, 0)
        except Exception as e:
            time.sleep(2 ** att + np.random.rand())
            err = str(e)
    return "ERR:" + err[:200], (0, 0)


def run_jobs(model, jobs, workers=4):
    """jobs: [(arm, day, prompt)] → 写缓存;返回 {(arm,day): (vol, pq, raw)}"""
    cache = load_cache(); cl = client(); todo = []
    for arm, day, pr in jobs:
        k = _key(model, arm, str(day.date()), pr)
        if k not in cache or cache[k].get("raw", "").startswith("ERR"):
            todo.append((k, arm, day, pr))
    print(f"[{model}] 任务 {len(jobs)};缓存命中 {len(jobs)-len(todo)};待调用 {len(todo)}", flush=True)
    done = [0]

    def work(t):
        k, arm, day, pr = t
        raw, (pt, ct) = call(cl, model, SYSTEM, pr)
        v, p = parse(raw)
        rec = dict(key=k, model=model, arm=arm, day=str(day.date()), vol=v, pq=p, raw=raw[:400], pt=pt, ct=ct)
        with _lock:
            with open(CACHE, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  …{done[0]}/{len(todo)}", flush=True)
        return rec
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, todo))
    cache = load_cache()
    return {(arm, str(day.date())): cache[_key(model, arm, str(day.date()), pr)] for arm, day, pr in jobs if _key(model, arm, str(day.date()), pr) in cache}


# ------------------------------------------------------------------ 评估
def week_boot(idx, fn, nb=4000, seed=0):
    wk = pd.Series(np.arange(len(idx)), index=idx).groupby([idx.isocalendar().year, idx.isocalendar().week]).apply(lambda s: s.to_numpy()).tolist()
    rg = np.random.default_rng(seed); o = []
    for _ in range(nb):
        pick = rg.integers(0, len(wk), len(wk)); o.append(fn(np.concatenate([wk[k] for k in pick])))
    return fn(np.arange(len(idx))), np.nanpercentile(o, 2.5), np.nanpercentile(o, 97.5)


def loo_r2(y, Z):
    """OLS 留一法 R²(相对 LOO 均值基线)"""
    Z = np.column_stack([np.ones(len(y)), Z]); H = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    res = (y - H @ y) / (1 - np.diag(H)); n = len(y)
    base = (y - (y.sum() - y) / (n - 1))
    return 1 - np.sum(res ** 2) / np.sum(base ** 2)


def evaluate(models, days_idx, S, meta):
    from scipy.stats import spearmanr
    cache = load_cache()
    Y, pred, thr = S["Y"], S["pred"], S["thr"]
    yA = pred[("A·ridge", "rv")]
    rows, inc = [], []
    ok = [d for d in days_idx if np.isfinite(yA.get(d, np.nan)) and np.isfinite(Y.loc[d, "y_rv"])]
    idx = pd.DatetimeIndex(ok)
    y = Y.loc[idx, "y_rv"].to_numpy(); ya = yA.loc[idx].to_numpy()
    low = (Y.loc[idx, "disp"] <= thr.loc[idx]).astype(int).to_numpy()
    print(f"评估样本 {len(idx)} 天 ({idx[0].date()}–{idx[-1].date()});低波标签占比 {low.mean():.2f};corr(ŷ_A, y_rv)={np.corrcoef(ya, y)[0,1]:.3f};ŷ_A 单独 LOO R²={loo_r2(y, ya):.3f}")
    naive = {"新闻条数(log)": np.log1p(meta.loc[idx, "n_items"].to_numpy()), "重要度≥2条数(log)": np.log1p(meta.loc[idx, "n_imp2"].to_numpy()),
             "总字数(log)": np.log1p(meta.loc[idx, "n_chars"].to_numpy())}
    scores = {}
    for mdl in models:
        for arm in ARMS:
            v = np.array([cache.get(_key(mdl, arm, str(d.date()), make_prompt(arm, meta.W[d], meta.M.loc[d])), {}).get("vol", np.nan) for d in idx])
            p = np.array([cache.get(_key(mdl, arm, str(d.date()), make_prompt(arm, meta.W[d], meta.M.loc[d])), {}).get("pq", np.nan) for d in idx])
            scores[(mdl, arm)] = (v, p)
    def block(nm, s_vol, s_pq):
        ok_ = np.isfinite(s_vol) & np.isfinite(y)
        if ok_.sum() < 60:
            return None
        r = {}
        f = lambda ii, a=s_vol: spearmanr(a[ii], y[ii])[0] if np.std(a[ii]) > 0 else np.nan
        pt, lo, hi = week_boot(idx, f, nb=1500); r.update(dict(rho=pt, rho_lo=lo, rho_hi=hi))
        fa = lambda ii, a=s_vol: LV.auc(1 - low[ii], a[ii])
        pt, lo, hi = week_boot(idx, fa, nb=1500); r.update(dict(AUC_高位移_vol=pt, auc_lo=lo, auc_hi=hi))
        if np.isfinite(s_pq).sum() > 60:
            r["AUC_高位移_(1−p_quiet)"] = LV.auc(1 - low, -np.nan_to_num(s_pq, nan=np.nanmedian(s_pq)))
        rg = np.random.default_rng(3); nul = np.array([spearmanr(rg.permutation(s_vol), y)[0] for _ in range(2000)])
        r["置换p(rho)"] = float(np.mean(nul >= r["rho"]))
        z = (s_vol - np.nanmean(s_vol)) / np.nanstd(s_vol)
        r["n"] = int(ok_.sum()); r["唯一取值数"] = int(len(np.unique(s_vol[ok_])))
        r["ΔR²(LOO) vs ŷ_A"] = loo_r2(y, np.column_stack([ya, z])) - loo_r2(y, ya)
        fi = lambda ii, z=z: loo_r2(y[ii], np.column_stack([ya[ii], z[ii]])) - loo_r2(y[ii], ya[ii]) if len(ii) > 30 else np.nan
        pt, lo, hi = week_boot(idx, fi, nb=800); r.update(dict(dR2_lo=lo, dR2_hi=hi))
        r["与ŷ_A相关"] = float(np.corrcoef(z, ya)[0, 1])
        mo = pd.Series(idx.month, index=idx)                                       # 月份固定效应稳健性:日期探针显示模型能从季节线索粗定位月份
        dm = lambda a: a - pd.Series(a, index=idx).groupby(mo.values).transform("mean").to_numpy()
        r["rho_月固定效应后"] = float(spearmanr(dm(s_vol), dm(y))[0]) if np.std(dm(s_vol)) > 0 else np.nan
        return dict(来源=nm, **r)
    for k, v_ in naive.items():
        rr = block(k, v_, np.full(len(idx), np.nan)); rows.append(rr) if rr else None
    for (mdl, arm), (v, p) in scores.items():
        rr = block(f"{mdl} / {arm}", v, p); rows.append(rr) if rr else None
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(R.round(3).to_string(index=False)); R.to_csv(TAB / "llm_news_vol_eval.csv", index=False)
    # 臂对比
    cmp_rows = []
    for mdl in models:
        for a_, b_ in (("N+M", "N"), ("N+M", "M"), ("N", "M")):
            va, vb = scores[(mdl, a_)][0], scores[(mdl, b_)][0]
            if np.isfinite(va).sum() < 60 or np.isfinite(vb).sum() < 60:
                continue
            f = lambda ii: LV.auc(1 - low[ii], np.nan_to_num(va[ii])) - LV.auc(1 - low[ii], np.nan_to_num(vb[ii]))
            pt, lo, hi = week_boot(idx, f, nb=1500)
            g = lambda ii: spearmanr(va[ii], y[ii])[0] - spearmanr(vb[ii], y[ii])[0]
            p2, lo2, hi2 = week_boot(idx, g, nb=1500)
            cmp_rows.append(dict(模型=mdl, 比较=f"{a_} − {b_}", ΔAUC=pt, lo=lo, hi=hi, Δrho=p2, rho_lo=lo2, rho_hi=hi2))
    C_ = pd.DataFrame(cmp_rows); print("\n臂对比:"); print(C_.round(3).to_string(index=False)); C_.to_csv(TAB / "llm_news_vol_arms.csv", index=False)
    return R, C_


class Meta:
    pass


def prepare():
    b = LV.load_bars("QQQ"); days = b.days
    news = load_news(); W = windows(days, news)
    M = macro_frame(days)
    meta = pd.DataFrame(index=pd.DatetimeIndex(sorted(W)))
    meta["n_items"] = [len(W[d]) for d in meta.index]; meta["n_imp2"] = [int((W[d].importance.fillna(0) >= 2).sum()) for d in meta.index]
    meta["n_chars"] = [int(W[d].text.str.len().sum()) for d in meta.index]
    mm = Meta(); mm.W = W; mm.M = M; mm.loc = meta.loc
    mm.index = meta.index
    return days, W, M, meta, mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true"); ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--probe"); ap.add_argument("--run"); ap.add_argument("--eval"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--models", default="llama3.1:8b,qwen3.6:35b")
    a = ap.parse_args()
    days, W, M, meta, mm = prepare()
    idx = mm.index
    print(f"窗口日数 {len(idx)}({idx[0].date()}–{idx[-1].date()});每窗新闻条数 中位 {meta.n_items.median():.0f},[min,max]=[{meta.n_items.min()},{meta.n_items.max()}];无新闻日 {(meta.n_items==0).sum()}")
    if a.prepare:
        d = idx[len(idx) // 2]
        print("\n--- 样例 N+M 提示词 ---"); print(make_prompt("N+M", W[d], M.loc[d])[:3500])
        raw = W[d].text.head(3).tolist(); print("\n--- 匿名化前后 ---")
        for t in raw: print("前:", t[:160]); print("后:", anonymize(t)[:160])
        print("\n宏观 NaN 占比:", M.loc[idx].isna().mean().round(2).to_dict())
        return
    if a.pilot:
        pick = [idx[5], idx[40], idx[80], idx[110]]
        for mdl in a.models.split(","):
            jobs = [(arm, d, make_prompt(arm, W[d], M.loc[d])) for d in pick for arm in ARMS]
            res = run_jobs(mdl, jobs, 3)
            for (arm, d), r in res.items():
                print(f"{mdl:14s} {arm:4s} {d}  vol={r['vol']} pq={r['pq']}  tok={r['pt']}/{r['ct']}  raw={r['raw'][:60]!r}")
        return
    if a.probe:
        rg = np.random.default_rng(0); pick = rg.choice(len(idx), 60, replace=False)
        cl = client(); rows = []
        def pr(i):
            d = idx[i]; txt = news_block(W[d])
            raw, _ = call(cl, a.probe, "Answer with JSON only.", txt + '\n\nWhen were these news items published? Return JSON only: {"year": <int>, "month": <1-12>}', 200)
            m = re.search(r'"year"\D+(\d{4}).*?"month"\D+(\d{1,2})', raw, re.S)
            return (d, int(m.group(1)), int(m.group(2))) if m else (d, np.nan, np.nan)
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(pr, pick))
        P_ = pd.DataFrame(rows, columns=["day", "gy", "gm"]).dropna()
        tm = P_.day.dt.year * 12 + P_.day.dt.month; gm = P_.gy * 12 + P_.gm
        from scipy.stats import spearmanr
        print(f"[日期探针 {a.probe}] n={len(P_)};猜 2026 年占比 {(P_.gy==2026).mean():.2f};猜测月份与真实月份 Spearman={spearmanr(tm, gm)[0]:.3f};|误差|≤1 月占比 {np.mean(np.abs(tm-gm)<=1):.2f};"
              f"猜测月份唯一值 {gm.nunique()}")
        return
    if a.run:
        for mdl in a.run.split(","):
            jobs = [(arm, d, make_prompt(arm, W[d], M.loc[d])) for d in idx for arm in ARMS]
            run_jobs(mdl, jobs, a.workers)
        return
    if a.eval:
        S = pd.read_pickle(OUT / "lowvol_study.pkl")
        evaluate(a.eval.split(","), idx, S, mm)


if __name__ == "__main__":
    main()
