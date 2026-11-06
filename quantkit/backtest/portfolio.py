"""Combining the core leg and the Sleeve (tearsheet v5 construction).

Who sizes what
--------------
* CORE (QQQ, held close-to-close):    target weight w_t from a fixed rule (quantkit.backtest.daily.core_weight). No fitted parameters.
* SLEEVE (TQQQ, intraday):            its long and short legs are scaled by the fitted PPP multipliers k_long / k_short
                                      on top of volatility targeting e_t = min(4, 2% / sigma14). The PPP never sizes the core.
* v5 PORTFOLIO LAYER (this file):     three fixed, non-fitted rules that act on BOTH legs afterwards:
    1. risk match   c_t = clip(vol252(reference portfolio) / vol252(w*r_core + rho*sleeve), 0.5, 1.5), lagged one day
                    Raising the Sleeve weight to rho would raise portfolio risk; c_t scales core and Sleeve down so that
                    total risk stays at the reference level. This is why the core weight falls when rho > 1.
    2. buying power f_t = min(1, cap / (c*w + e*k/3 * c*rho))  scales both legs when the day's worst-case notional
                    (core + Sleeve TQQQ notional) would exceed the margin guardrail `cap`.
    3. no-trade band the executed core weight only moves when the target differs by >= band (or the cap is breached).
  Sleeve notional per unit of capital is e*k/3 because TQQQ is 3x.

PIT warning
-----------
In the original tearsheet_v5, k in step 2 was the day's realised max(k_long, k_short), which depends on features known
only at 10:00, yet f_t was applied to the whole day's core return (including the overnight gap and 09:30-10:00).
That is a small look-ahead (PR #2 found it; effect on OOS Sharpe about -0.02 to -0.03 once removed).
Use pit_cap=True (k_upper = 2, the model's maximum) for anything that must be implementable at the open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAP_DEFAULT = 0.85 * 1_513_400 / 500_000     # 0.85 x buying power / CAPITAL_BASE = 2.57
RHO_DEFAULT = 1.5
BAND_DEFAULT = 0.10
K_MAX = 2.0


def build_v5(core_w, sleeve, e_vt, k_used, core_ret, ref_port, rho: float = RHO_DEFAULT, cap: float = CAP_DEFAULT, band: float = BAND_DEFAULT,
             use_ratio: bool = True, use_band: bool = True, use_cap: bool = True, pit_cap: bool = False, core_cost: float = 2e-4) -> dict:
    """All inputs are arrays aligned on trading days.
    core_w   target core weight (PIT)           sleeve   Sleeve daily return with PPP multipliers already applied (v4 Sleeve)
    e_vt     vol-target leverage min(4, 2%/sigma14)   k_used   max(k_long, k_short) of the day (ignored if pit_cap)
    core_ret core asset daily return            ref_port reference portfolio (v4) whose risk is matched
    Returns dict(port, core_w, sleeve_mult, usage_pre, usage_post, f, c)."""
    core_w, sleeve, e_vt, core_ret = (np.asarray(a, float) for a in (core_w, sleeve, e_vt, core_ret)); n = len(core_ret)
    k = np.full(n, K_MAX) if pit_cap else np.asarray(k_used, float)
    if use_ratio:
        mix = pd.Series(core_w * core_ret + rho * sleeve)
        c = (pd.Series(ref_port).rolling(252, min_periods=126).std() / mix.rolling(252, min_periods=126).std()).shift(1).clip(0.5, 1.5).fillna(1.0).to_numpy()
    else:
        c = np.ones(n); rho = 1.0
    tgt = c * core_w; sm = c * rho; slv = e_vt * k / 3.0; usage_pre = tgt + slv * sm
    f = np.minimum(1.0, cap / np.maximum(usage_pre, 1e-12)) if use_cap else np.ones(n); tgt = tgt * f; sm = sm * f
    w = np.empty(n); cur = tgt[0]
    for i in range(n):
        if (not use_band) or abs(tgt[i] - cur) >= band or (use_cap and cur + slv[i] * sm[i] > cap + 1e-12):
            cur = tgt[i]
        w[i] = cur
    cost = core_cost * np.abs(np.diff(np.r_[w[0], w]))
    return dict(port=w * core_ret - cost + sm * sleeve, core_w=w, sleeve_mult=sm, usage_pre=usage_pre, usage_post=w + slv * sm, f=f, c=c)
