"""本库只 vendor 了 intraday 引擎。157 个日频因子的 REGISTRY 属于 etf_timing 全量工程，未随本库交付。"""
def __getattr__(name):
    raise ImportError(f'etft.factors.{name} 未随本库交付（仅 vendor 了 intraday 引擎），见 docs/FULL_RESEARCH_REPORT.md')
