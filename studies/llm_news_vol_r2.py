"""LLM 新闻→波动 第二轮 —— 见 docs/PROPOSAL_llm_news_vol.md「第二轮」。
  python llm_news_vol_r2.py --variant v1 --split dev  --models llama3.1:8b,qwen3.6:35b   # 迭代阶段只调用开发集
  python llm_news_vol_r2.py --variant v1 --split hold --models ...                      # 冻结后才允许调用(留出集)
  python llm_news_vol_r2.py --variant v1 --eval dev|hold|all --models ...
环境变量 SOCLAAS_API_KEY / SOCLAAS_BASE_URL。"""
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
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_news_vol_study as L0
import lowvol_avoidance_study as LV

OUT, TAB = L0.OUT, L0.TAB
CACHE = OUT / "llm_cache_r2.jsonl"
DEV_END = pd.Timestamp("2026-07-31")
FIELDS = ["already_priced", "pending_event", "escalation", "surprise", "vol_vs_baseline"]
_lock = threading.Lock()

SYSTEM = ("You are an equity-index volatility analyst. You separate (a) what has ALREADY happened and is priced in by 10:00 from "
          "(b) what is STILL TO COME later in the session. Answer with a single JSON object only.")

TASKS = {
 "v1": ("Context: it is 30 minutes after the U.S. open. The news above covers the period from the previous close until now (dates and index levels removed). "
        "A quantitative model that already uses VIX, recent realized volatility and dealer gamma expects the volatility of the REST of today's session "
        "(10:00 until the close) to be at the {pct}th percentile of typical days. You must NOT re-estimate that baseline. "
        "Judge ONLY whether the NEWS implies that the rest-of-session volatility will be higher or lower than that baseline.\n"
        "Think about: shocks that already occurred before 10:00 are mostly priced and often calm down; events scheduled or developing AFTER 10:00 "
        "(data releases, central-bank decisions or speeches, earnings, deadlines, negotiations) add volatility; escalation vs de-escalation of geopolitical or policy risk.\n"
        'Return JSON only with integer fields: {"already_priced": <0-10, how much of the major news shock has already been absorbed by 10:00>, '
        '"pending_event": <0 or 1, is there a clearly identified catalyst still to come later today>, '
        '"escalation": <-2..2, -2 strong de-escalation, 2 strong escalation of risk>, '
        '"surprise": <0-10, size of surprise relative to expectations>, '
        '"vol_vs_baseline": <-2..2, -2 rest-of-session much calmer than baseline, 0 as expected, 2 much more volatile than baseline>}'),
 "v2": ("Context: it is 30 minutes after the U.S. open. The news above covers the period from the previous close until now (dates and index levels removed). "
        "A quantitative model that already uses VIX, recent realized volatility and dealer gamma expects the volatility of the REST of today's session "
        "(10:00 until the close) to be at the {pct}th percentile of typical days. Do NOT re-estimate that baseline; judge ONLY what the NEWS adds.\n"
        "Scoring rules (be conservative; most days should score at the neutral value):\n"
        "- already_priced (0-10): 10 = the main shock in the news was fully absorbed before 10:00 (e.g. a reversal already completed); 0 = the shock is still unfolding or only just broke.\n"
        "- pending_event (0 or 1): 1 ONLY if the news explicitly names a specific catalyst that happens AFTER 10:00 today (a scheduled data release, a central-bank decision or press conference, "
        "a named speech, a deadline, a vote, a court ruling, a major earnings report after the close or at midday, a named negotiation round). Generic 'ongoing tensions' is 0. Typical frequency: about 1 day in 4.\n"
        "- escalation (-2..2): direction of change in geopolitical/policy risk during this window; 0 = no real change (typical).\n"
        "- surprise (0-10): how unexpected the news is versus consensus; 3 = ordinary day, 6+ = clearly unexpected, 9-10 = historic.\n"
        "- vol_vs_baseline (-2..2): 0 = news adds nothing beyond the baseline (the typical answer). +1/+2 only if there is a specific unpriced catalyst still to come; -1/-2 only if the biggest news is clearly resolved and nothing else is scheduled.\n"
        'Return JSON only with integer fields: {"already_priced":..,"pending_event":..,"escalation":..,"surprise":..,"vol_vs_baseline":..}'),
}


def pct_baseline(S):
    """ŷ_A 的因果分位:t 日预测在『此前所有 OOS 预测』中的位置(百分比,取整)。"""
    yA = S["pred"][("A·ridge", "rv")].dropna(); v = yA.to_numpy(); out = {}
    for i, d in enumerate(yA.index):
        out[d] = float(np.mean(v[:i] < v[i]) * 100) if i >= 60 else np.nan
    return pd.Series(out)


def make_prompt(variant, w, pct):
    return L0.news_block(w) + "\n\n" + TASKS[variant].replace("{pct}", str(int(round(pct)) if np.isfinite(pct) else 50))


def key(model, variant, day, prompt):
    return hashlib.sha1(f"{variant}|{model}|{day}|{prompt}|{SYSTEM}".encode()).hexdigest()


def load_cache():
    c = {}
    if CACHE.exists():
        for ln in CACHE.read_text().splitlines():
            try:
                r = json.loads(ln); c[r["key"]] = r
            except Exception:
                pass
    return c


def parse(txt):
    m = re.findall(r"\{.*?\}", txt, re.S)
    for blob in reversed(m):
        try:
            j = json.loads(blob)
            if "vol_vs_baseline" in j:
                return {f: float(j.get(f, np.nan)) for f in FIELDS}
        except Exception:
            continue
    out = {}
    for f in FIELDS:
        a = re.search(rf'"?{f}"?\W+(-?\d+(?:\.\d+)?)', txt)
        out[f] = float(a.group(1)) if a else np.nan
    return out


def run(model, variant, days, W, pct, workers=3):
    cache = load_cache(); cl = L0.client(); todo = []
    mt = 3000 if model.startswith("qwen3.8") else 1200
    for d in days:
        pr = make_prompt(variant, W[d], pct.get(d, np.nan)); k = key(model, variant, str(d.date()), pr)
        if k not in cache or cache[k].get("raw", "").startswith("ERR"):
            todo.append((k, d, pr))
    print(f"[{model}/{variant}] 天数 {len(days)};缓存命中 {len(days)-len(todo)};待调用 {len(todo)}", flush=True)

    def work(t):
        k, d, pr = t
        raw, (pt, ct) = L0.call(cl, model, SYSTEM, pr, mt)
        rec = dict(key=k, model=model, variant=variant, day=str(d.date()), raw=raw[-1500:], pt=pt, ct=ct, **parse(raw))
        with _lock:
            with open(CACHE, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, todo))


def table(model, variant, days, W, pct):
    c = load_cache(); rows = []
    for d in days:
        pr = make_prompt(variant, W[d], pct.get(d, np.nan)); r = c.get(key(model, variant, str(d.date()), pr))
        rows.append({f: (r[f] if r else np.nan) for f in FIELDS})
    return pd.DataFrame(rows, index=days)


def evaluate(models, variant, days, W, pct, S, label):
    from scipy.stats import spearmanr
    Y, yA = S["Y"], S["pred"][("A·ridge", "rv")]
    ok = [d for d in days if np.isfinite(yA.get(d, np.nan))]
    idx = pd.DatetimeIndex(ok); y = Y.loc[idx, "y_rv"].to_numpy(); ya = yA.loc[idx].to_numpy(); res = y - ya
    print(f"\n[{label}] n={len(idx)};残差 y−ŷ_A 的 sd={res.std():.3f};corr(ŷ_A,y)={np.corrcoef(ya,y)[0,1]:.3f};ŷ_A LOO R²={L0.loo_r2(y, ya):.3f}")
    rows = []
    for m in models:
        T_ = table(m, variant, idx, W, pct)
        parse_rate = float(T_.vol_vs_baseline.notna().mean())
        for f in FIELDS + ["composite"]:
            if f == "composite":
                z = lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1)
                s = (z(T_.vol_vs_baseline) + z(T_.pending_event) + z(T_.escalation) - z(T_.already_priced)).to_numpy()
            else:
                s = T_[f].to_numpy()
            okk = np.isfinite(s)
            if okk.sum() < 30 or np.nanstd(s) == 0:
                rows.append(dict(模型=m, 字段=f, n=int(okk.sum()), 解析率=parse_rate, rho_残差=np.nan)); continue
            fr = lambda ii, s=s: spearmanr(s[ii], res[ii], nan_policy="omit")[0]
            pt, lo, hi = L0.week_boot(idx, fr, nb=1500)
            zs = (np.nan_to_num(s, nan=np.nanmean(s)) - np.nanmean(s)) / np.nanstd(s)
            d_r2 = L0.loo_r2(y, np.column_stack([ya, zs])) - L0.loo_r2(y, ya)
            fi = lambda ii, zs=zs: L0.loo_r2(y[ii], np.column_stack([ya[ii], zs[ii]])) - L0.loo_r2(y[ii], ya[ii]) if len(ii) > 20 else np.nan
            _, lo2, hi2 = L0.week_boot(idx, fi, nb=600)
            rows.append(dict(模型=m, 字段=f, n=int(okk.sum()), 解析率=parse_rate, 唯一值=int(len(np.unique(s[okk]))), rho_残差=pt, lo=lo, hi=hi, ΔR2_LOO=d_r2, dR2_lo=lo2, dR2_hi=hi2,
                             与ŷ_A相关=float(spearmanr(s, ya, nan_policy="omit")[0])))
    R = pd.DataFrame(rows); pd.set_option("display.width", 220)
    print(R.round(3).to_string(index=False))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1"); ap.add_argument("--split", choices=["dev", "hold"]); ap.add_argument("--eval", choices=["dev", "hold", "all"])
    ap.add_argument("--models", default="llama3.1:8b,qwen3.6:35b"); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    days, W, M, meta, mm = L0.prepare(); idx = mm.index
    S = pd.read_pickle(OUT / "lowvol_study.pkl"); pct = pct_baseline(S)
    dev = idx[idx <= DEV_END]; hold = idx[idx > DEV_END]
    if a.show:
        d = dev[a.show]; print(make_prompt(a.variant, W[d], pct.get(d, np.nan))[-1800:]); return
    models = a.models.split(",")
    if a.split:
        use = dev if a.split == "dev" else hold
        for m in models:
            run(m, a.variant, use, W, pct, a.workers)
    if a.eval:
        use = {"dev": dev, "hold": hold, "all": idx}[a.eval]
        R = evaluate(models, a.variant, use, W, pct, S, f"{a.variant}/{a.eval}")
        R.to_csv(TAB / f"llm_news_vol_r2_{a.variant}_{a.eval}.csv", index=False)


if __name__ == "__main__":
    main()
