"""quantkit: reusable research infrastructure for the QQQ core + noise-area Sleeve project.

    quantkit.data        fetch / load / checksum raw data (Yahoo, Cboe, Ken French, SqueezeMetrics, moomoo, mdrec)
    quantkit.backtest    daily core-leg primitives, the v5 portfolio combiner, thin adapters to the minute engine
    quantkit.stats       performance panel, block bootstrap, HAC tests, PSR/DSR/PBO, PSI, monotonicity, attribution, CV
    quantkit.execution   order-book walking and slippage tables

Conventions (same as the research scripts):
  * returns are simple daily returns, indexed by trading day; annualisation uses 252
  * a signal dated t is only allowed to use information up to t-1 (daily) or the decision bar (intraday)
  * data root = $ETFT_DATA (default <repo>/data); raw files live under <root>/raw
See docs/RESEARCH_GUIDE.md.
"""
__version__ = "0.1.0"
