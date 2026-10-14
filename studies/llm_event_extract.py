"""LLM 逐条新闻结构化事件抽取 —— docs/PROPOSAL_llm_news_vol.md「第三轮」。
  python llm_event_extract.py --run MODEL     # 抽取并缓存(批 20 条)
  python llm_event_extract.py --eval          # 聚合、拟合(开发集)、留出集评估
环境变量 SOCLAAS_API_KEY / SOCLAAS_BASE_URL。"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, threading, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_news_vol_study as L0
import llm_news_vol_r2 as R2

OUT, TAB = L0.OUT, L0.TAB
CACHE = OUT / "llm_extract_cache.jsonl"
ETYPES = ["central_bank", "macro_data", "earnings", "geopolitics", "policy_trade", "mkt_wrap", "corporate", "other"]
TIMINGS = ["past", "today_later", "future", "none"]
MAXN, BATCH = 100, 20
_lock = threading.Lock()
SYSTEM = "You are a precise financial-news annotator. Output a single JSON array and nothing else."
EXTRACT_V1 = ("Annotate each numbered news item (dates and index levels were removed). For every item return an object with:\n"
              '"i": the item number,\n'
              '"etype": one of central_bank (central bank decisions or officials\' remarks), macro_data (economic data releases or expectations), earnings (corporate earnings/guidance), '
              "geopolitics (war, conflict, sanctions, diplomacy), policy_trade (tariffs, regulation, fiscal policy, legislation), mkt_wrap (market close or pre-market summaries), corporate (M&A, IPO, single-company events), other;\n"
              '"timing": past (the described event already happened or was already released), today_later (the item explicitly says something will happen or be released later today / tonight / during the session), '
              "future (a later day), none (no event timing);\n"
              '"severity": 0 routine, 1 noteworthy, 2 a major event that could cause large market swings.\n'
              "Return ONLY a JSON array like [{\"i\":1,\"etype\":\"macro_data\",\"timing\":\"past\",\"severity\":0}, ...] with one object per item.\n\nITEMS:\n")


def items_for_day(w):
    if not len(w): return []
    w = w.assign(n=w.importance.fillna(0)).sort_values(["n", "ts"], ascending=[False, True]).head(MAXN).sort_values("ts")
    return [L0.anonymize(t)[:400] for t in w.text]


def key(model, day, b, txt):
    return hashlib.sha1(f"EXTRACT_V1|{model}|{day}|{b}|{txt}".encode()).hexdigest()


def load_cache():
    c = {}
    if CACHE.exists():
        for ln in CACHE.read_text().splitlines():
            try: r = json.loads(ln); c[r["key"]] = r
            except Exception: pass
    return c


def parse_arr(txt):
    m = re.search(r"\[.*\]", txt, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception:
        objs = re.findall(r"\{[^{}]*\}", m.group(0))
        out = []
        for o in objs:
            try: out.append(json.loads(o))
            except Exception: pass
        return out or None


def run(model, days, W, workers=3):
    cache = load_cache(); cl = L0.client(); jobs = []
    for d in days:
        its = items_for_day(W[d])
        for b in range(0, len(its), BATCH):
            chunk = its[b:b + BATCH]; txt = "\n".join(f"{j+1}. {t}" for j, t in enumerate(chunk))
            k = key(model, str(d.date()), b, txt)
            if k not in cache or cache[k].get("raw", "").startswith("ERR"):
                jobs.append((k, d, b, len(chunk), txt))
    print(f"[{model}] 批次待调用 {len(jobs)}", flush=True)
    def work(t):
        k, d, b, n, txt = t
        raw, (pt, ct) = L0.call(cl, model, SYSTEM, EXTRACT_V1 + txt, 1500)
        arr = parse_arr(raw)
        rec = dict(key=k, model=model, day=str(d.date()), b=b, n=n, raw=raw[:200] if arr else raw[:600], pt=pt, ct=ct, arr=arr)
        with _lock:
            with open(CACHE, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with ThreadPoolExecutor(workers) as ex: list(ex.map(work, jobs))


def aggregate(model, days, W):
    c = load_cache(); rows = []; kap = []
    for d in days:
        its = items_for_day(W[d]); recs = []
        for b in range(0, len(its), BATCH):
            chunk = its[b:b + BATCH]; txt = "\n".join(f"{j+1}. {t}" for j, t in enumerate(chunk)); r = c.get(key(model, str(d.date()), b, txt))
            if r and r.get("arr"):
                for o in r["arr"]:
                    if isinstance(o, dict): recs.append(o)
        df = pd.DataFrame(recs) if recs else pd.DataFrame(columns=["etype", "timing", "severity"])
        for col in ("etype", "timing"): df[col] = df.get(col, pd.Series(dtype=str)).astype(str)
        sev = pd.to_numeric(df.get("severity", pd.Series(dtype=float)), errors="coerce").fillna(0)
        rows.append(dict(day=d, n_parsed=len(df), n_items=len(its), n_today_later=int((df.timing == "today_later").sum()), n_central_bank=int((df.etype == "central_bank").sum()),
                         n_macro_data=int((df.etype == "macro_data").sum()), n_earnings=int((df.etype == "earnings").sum()), n_geopolitics=int((df.etype == "geopolitics").sum()),
                         n_policy_trade=int((df.etype == "policy_trade").sum()), max_severity=float(sev.max()) if len(sev) else 0.0, n_sev2=int((sev >= 2).sum())))
    A = pd.DataFrame(rows).set_index("day"); A["any_today_later"] = (A.n_today_later > 0).astype(float)
    for c_ in ("n_today_later", "n_central_bank", "n_macro_data", "n_earnings", "n_geopolitics", "n_policy_trade", "n_sev2"): A[c_] = np.log1p(A[c_])
    return A


FEATS = ["n_today_later", "any_today_later", "n_central_bank", "n_macro_data", "n_earnings", "n_geopolitics", "n_policy_trade", "max_severity", "n_sev2"]


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b); cats = sorted(set(a) | set(b)); po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats); return (po - pe) / (1 - pe) if pe < 1 else np.nan


def evaluate(models, days, W, S):
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    Y, yA = S["Y"], S["pred"][("A·ridge", "rv")]
    idx = pd.DatetimeIndex([d for d in days if np.isfinite(yA.get(d, np.nan))]); res = (Y.loc[idx, "y_rv"] - yA.loc[idx])
    dev = idx[idx <= R2.DEV_END]; hold = idx[idx > R2.DEV_END]
    print(f"样本 {len(idx)} 天;开发 {len(dev)} / 留出 {len(hold)};残差 sd(dev)={res[dev].std():.3f}, sd(hold)={res[hold].std():.3f}")
    aggs = {m: aggregate(m, idx, W) for m in models}
    for m, A in aggs.items():
        print(f"{m}: 解析覆盖率 {A.n_parsed.sum()/max(A.n_items.sum(),1):.3f};today_later>0 的天数占比 {(A.any_today_later>0).mean():.2f};事件类型总计 {dict(A[['n_central_bank','n_macro_data','n_earnings','n_geopolitics','n_policy_trade']].apply(np.expm1).sum().round(0))}")
    # 抽取一致性(逐日计数相关,代替逐条 κ,因为两模型批次内编号可能错位)
    if len(models) == 2:
        a, b = aggs[models[0]], aggs[models[1]]
        print("两模型日级计数的 Spearman(一致性诊断):", {f: round(float(spearmanr(a[f], b[f])[0]), 2) for f in FEATS})
    rows = []; out = {}
    for m, A in aggs.items():
        X = A.reindex(idx)[FEATS].fillna(0.0)
        uni = {f: spearmanr(X.loc[dev, f], res[dev])[0] for f in FEATS}
        mu, sd = X.loc[dev].mean(), X.loc[dev].std().replace(0, 1)
        mdl = Ridge(alpha=10.0).fit(((X.loc[dev] - mu) / sd).to_numpy(), res[dev].to_numpy())
        coef = pd.Series(mdl.coef_, index=FEATS)
        ph = mdl.predict(((X.loc[hold] - mu) / sd).to_numpy()); yh = res[hold].to_numpy()
        base = res[dev].mean()
        r2 = 1 - np.sum((yh - ph) ** 2) / np.sum((yh - base) ** 2)
        rho = spearmanr(ph, yh)[0]
        fr = lambda ii: spearmanr(ph[ii], yh[ii])[0]
        pt, lo, hi = L0.week_boot(hold, fr, nb=2000)
        q = np.quantile(ph, [1 / 3, 2 / 3]); top, bot = yh[ph >= q[1]].mean(), yh[ph <= q[0]].mean()
        # 开发集 LOO R²
        Xs = ((X.loc[dev] - mu) / sd).to_numpy(); yd = res[dev].to_numpy(); loo = L0.loo_r2(yd, Xs)
        print(f"\n[{m}] 开发集单变量 Spearman(特征, 残差):", {f: round(float(v), 2) for f, v in uni.items()})
        print(f"[{m}] 开发集岭系数:", coef.round(3).to_dict(), f"; 开发集 LOO R²(9 特征)={loo:.3f}")
        print(f"[{m}] 留出集(n={len(hold)}):OOS R²={r2:+.3f};Spearman(预测,实际残差)={rho:+.3f} [{lo:+.2f},{hi:+.2f}];预测最高三分位残差 {top:+.3f} vs 最低 {bot:+.3f}")
        rows.append(dict(模型=m, 留出OOS_R2=r2, 留出rho=rho, rho_lo=lo, rho_hi=hi, 高三分位残差=top, 低三分位残差=bot, 开发LOO_R2=loo))
        for f in FEATS: rows[-1][f"dev_rho_{f}"] = uni[f]
    R = pd.DataFrame(rows); R.to_csv(TAB / "llm_extract_eval.csv", index=False); return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run"); ap.add_argument("--eval", action="store_true"); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--models", default="qwen3.6:35b,llama3.1:8b"); a = ap.parse_args()
    days, W, M, meta, mm = L0.prepare(); idx = mm.index
    if a.run:
        for m in a.run.split(","): run(m, idx, W, a.workers)
    if a.eval:
        evaluate(a.models.split(","), idx, pd.read_pickle(OUT / "lowvol_study.pkl"), ) if False else evaluate(a.models.split(","), idx, W, pd.read_pickle(OUT / "lowvol_study.pkl"))


if __name__ == "__main__":
    main()
