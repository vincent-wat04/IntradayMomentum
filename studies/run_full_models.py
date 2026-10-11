"""Full-sample model study (2019-01 .. 2026-10): rolling Ridge / ElasticNet, RAVEN-lite and the
paper-architecture RAVEN, with and without the macro factors that pass a PRETEST-ONLY filter.

Re-uses the repository's model code unchanged (multifrequency_models.screen_factors / run_one,
raven_paper_model.fit_predict); only the sample split, the early-close calendar and the candidate
matrix are configured here.

Protocol (fixed before running)
  * TEST_START = 2023-01-03: screening, macro pre-filter and every hyper-parameter use only labels
    that EXITED before the first test signal (2019-01 .. 2022-12 = 4 pretest years incl. the 2020
    crash and the 2022 bear market).  The original repo used 2025-06-02 on 397 days.
  * Feature set A ("intraday"): the repo's 168 orthogonalised intraday/price candidates.
  * Feature set B ("intraday+macro"): A + macro (name, K) columns (orthogonalised, K in 1/5/20/60)
    that pass the macro pre-filter for that (horizon, execution):
      - per pretest calendar year (2019..2022), Pearson IC with matured labels only;
      - same sign in all 4 years for BOTH SPY and QQQ, and mean |IC| >= 0.05.
    screen_factors then chooses <= 4 features from the union exactly as before.
  * Models (pre-set, unchanged from the repo): Ridge alpha 10 / ElasticNet alpha .05 l1 .5 on
    60 / 120 / expanding windows, daily refit; RAVEN-lite expanding; RAVEN (paper architecture)
    retrained every `--raven-retrain` signals with `--raven-epochs`.

    python run_full_models.py --stage linear           # Ridge / EN / RAVEN-lite, sets A and B
    python run_full_models.py --stage raven --workers 8
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import multifrequency_models as MF
from index_timing_framework import load_market, orthogonalize_past_k
from macro_factor_study import load_macro_panel
from raven_lite_timing import backtest, measure
from run_full_extended import config_from

TEST_START = "2023-01-03"
MF.TEST_START = TEST_START
DATA, ETFT, OUT = Path("data"), Path("../etf_timing"), Path("outputs/full")
MACRO_K = (1, 5, 20, 60)
PRETEST_YEARS = ("2019", "2020", "2021", "2022")


def macro_matrix(sym, panel, cfg):
    X, _, _ = load_macro_panel(ETFT, sym, panel.dates)
    cols = {}
    for name in X.columns:
        for k in MACRO_K:
            raw = X[name].rolling(k, min_periods=k).mean()
            cols[("M:" + name, k)] = orthogonalize_past_k(raw, panel, k, cfg).reindex(panel.dates)
    return pd.DataFrame(cols, index=panel.dates)


def pretest_yearly_ic(matrix, y, maturity, dates, split):
    """IC per pretest year using only labels matured before the split."""
    yrs = np.array([d[:4] for d in dates])
    ok_lab = (maturity < split) & np.isfinite(y)
    out = {}
    for col in matrix.columns:
        x = matrix[col].to_numpy(float)
        ics = []
        for yr in PRETEST_YEARS:
            m = ok_lab & (yrs == yr) & np.isfinite(x)
            ics.append(np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 40 and np.std(x[m]) > 0 else np.nan)
        out[col] = ics
    return out


def macro_filter(per_symbol_ics):
    passed = []
    cols = per_symbol_ics["SPY"].keys()
    for c in cols:
        a = np.array(per_symbol_ics["SPY"][c] + per_symbol_ics["QQQ"][c], float)
        if not np.isfinite(a).all():
            continue
        if (np.all(a > 0) or np.all(a < 0)) and abs(a.mean()) >= 0.05:
            passed.append(c)
    return passed


def prepare():
    cfg = config_from(DATA)
    P = {}
    for sym in ("SPY", "QQQ"):
        path = str(DATA / f"{sym}_5min.csv")
        panel = load_market(path, cfg)
        ca, cm, cv = (OUT / "cache" / f"{sym}_{n}.pkl" for n in ("A", "M", "valid"))
        if ca.exists():
            mA, mM, valid_days = pd.read_pickle(ca), pd.read_pickle(cm), pd.read_pickle(cv)
        else:
            mA, valid_days = MF.factor_matrix(path, panel, cfg)
            mM = macro_matrix(sym, panel, cfg)
            ca.parent.mkdir(parents=True, exist_ok=True)
            mA.to_pickle(ca); mM.to_pickle(cm); pd.Series(list(valid_days)).to_pickle(cv)
            valid_days = pd.Series(list(valid_days))
        P[sym] = dict(path=path, panel=panel, A=mA, M=mM, valid=set(valid_days))
    return cfg, P


def run_linear(horizons=MF.HORIZONS):
    cfg, P = prepare()
    results, filt = [], {}
    split = {s: int(np.searchsorted(P[s]["panel"].dates, TEST_START)) for s in P}
    for h in horizons:
        for ex in MF.EXECUTIONS:
            ics = {}
            labs = {}
            for s in P:
                y, mat = MF.target(P[s]["panel"], h, ex)
                labs[s] = (y, mat)
                ics[s] = pretest_yearly_ic(P[s]["M"], y, mat, list(P[s]["panel"].dates), split[s])
            passed = macro_filter(ics)
            filt[f"{h}_{ex}"] = {"passed": [f"{c[0][2:]}:{c[1]}" for c in passed],
                                 "ic": {f"{c[0][2:]}:{c[1]}": {s: [round(v, 4) for v in ics[s][c]] for s in P} for c in passed}}
            print(f"h={h} {ex}: macro passing pretest filter: {len(passed)}", filt[f'{h}_{ex}']['passed'][:12], flush=True)
            for s in P:
                y, mat = labs[s]
                for fs, matrix in (("A_intraday", P[s]["A"]),
                                   ("B_intraday+macro", pd.concat([P[s]["A"], P[s]["M"][passed]], axis=1) if passed else None)):
                    if matrix is None:
                        continue
                    sel, scr = MF.screen_factors(matrix, y, mat, split[s])
                    for model in ("ridge", "elastic_net", "raven_lite"):
                        for win in ((60, 120, None) if model != "raven_lite" else (None,)):
                            t0 = time.time()
                            r = MF.run_one(P[s]["path"], s, cfg, h, ex, model, win, matrix, P[s]["valid"], sel, scr, P[s]["panel"])
                            r["feature_set"] = fs
                            results.append(r)
                            print(f"  {s} {fs:17s} {model:11s} {str(win):5s} {r['status']} n={r.get('n_test')} "
                                  f"IC={r['metric']['pearson'] if r.get('metric') else None} "
                                  f"feat={[f['name']+':'+str(f['k']) for f in r.get('selected',[])]} {time.time()-t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_".join(map(str, horizons))
    (OUT / f"macro_pretest_filter_h{tag}.json").write_text(json.dumps(filt, indent=1))
    (OUT / f"models_linear_h{tag}.json").write_text(json.dumps(results, allow_nan=False, default=float))


# ----------------------------------------------------------------------------- full RAVEN
_R = {}


def _rinit():
    cfg, P = prepare()
    filt = {}
    for f in OUT.glob("macro_pretest_filter_h*.json"):
        filt.update(json.loads(f.read_text()))
    _R.update(cfg=cfg, P=P, filt=filt)


def _raven_job(job):
    import raven_paper_model as RP
    s, h, ex, fs, epochs, retrain = job
    P = _R["P"][s]; panel = P["panel"]
    passed = [("M:" + c.split(":")[0], int(c.split(":")[1])) for c in _R["filt"][f"{h}_{ex}"]["passed"]]
    matrix = P["A"] if fs == "A_intraday" else pd.concat([P["A"], P["M"][passed]], axis=1)
    y, maturity = MF.target(panel, h, ex)
    split = int(np.searchsorted(panel.dates, TEST_START))
    selected, diag = MF.screen_factors(matrix, y, maturity, split)
    base = dict(symbol=s, horizon=h, execution=ex, model="raven_full", training_window="expanding", feature_set=fs)
    if not selected:
        return {**base, "status": "no_stable_features"}
    days = np.array([i for i in range(len(y)) if panel.dates[i] in P["valid"]], int)
    windows, pos = MF.windows_for(matrix, selected, days, length=64)
    test_idx = np.where((pos >= split) & np.isfinite(y[pos]))[0]
    preds, truth, tpos, counts, prefixes = [], [], [], [], []
    fitted_end, cache, fit_count, params = -1, {}, 0, None
    t0 = time.time()
    for ix in test_idx:
        t = int(pos[ix])
        if fitted_end < 0 or t - fitted_end >= retrain:
            train = np.where((pos < t) & (maturity[pos] <= t) & np.isfinite(y[pos]))[0]
            if len(train) < 40:
                continue
            fitted_end = t
            blk = test_idx[(test_idx >= ix)][:retrain + 5]
            pb, pf, params = RP.fit_predict(windows[train], y[pos[train]], windows[blk], epochs=epochs, seed=7 + t)
            cache = {int(pos[z]): (float(v), p.tolist()) for z, v, p in zip(blk, pb, pf)}
            fit_count = len(train)
        if t not in cache:
            continue
        v, p = cache[t]
        preds.append(v); truth.append(y[t]); tpos.append(t); counts.append(fit_count); prefixes.append(p)
    pred, truth, tpos = np.asarray(preds), np.asarray(truth), np.asarray(tpos)
    met = MF.model_diagnostics(truth, pred, tpos, h, ex)
    return {**base, "status": "ok", "selected": selected, "n_test": len(pred), "n_train_min": min(counts),
            "n_train_max": max(counts), "parameters": params, "epochs": epochs, "retrain_every": retrain,
            "prefix_means": np.mean(prefixes, axis=0).tolist(), "metric": met,
            "daily_long_cash": backtest(panel, tpos, pred, ex), "minutes": (time.time() - t0) / 60,
            "first_signal": panel.dates[tpos[0]], "last_signal": panel.dates[tpos[-1]],
            "predictions": [dict(signal_day=panel.dates[t], forecast=float(v), realized=float(q))
                            for t, v, q in zip(tpos, pred, truth)]}


def run_raven(workers, epochs, retrain, horizons=MF.HORIZONS, executions=MF.EXECUTIONS):
    jobs = [(s, h, ex, fs, epochs, retrain) for fs in ("A_intraday", "B_intraday+macro")
            for h in horizons for ex in executions for s in ("SPY", "QQQ")]
    fn = OUT / f"models_raven_full_h{'_'.join(map(str, horizons))}_{'_'.join(executions)}.json"
    out = []
    with Pool(workers, initializer=_rinit) as p:
        for r in p.imap_unordered(_raven_job, jobs):
            out.append(r)
            print(r["symbol"], r["horizon"], r["execution"], r["feature_set"], r["status"], r.get("n_test"),
                  r.get("metric", {}).get("pearson") if r.get("metric") else None, f"{r.get('minutes', 0):.1f}min", flush=True)
            fn.write_text(json.dumps(out, allow_nan=False, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["linear", "raven"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--raven-epochs", type=int, default=30)
    ap.add_argument("--raven-retrain", type=int, default=63)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(MF.HORIZONS))
    ap.add_argument("--executions", nargs="+", default=list(MF.EXECUTIONS))
    a = ap.parse_args()
    if a.stage == "linear":
        run_linear(tuple(a.horizons))
    else:
        run_raven(a.workers, a.raven_epochs, a.raven_retrain, tuple(a.horizons), tuple(a.executions))
