# 研究基础设施指南

给自己动手跑研究用。分三部分:仓库地图、`quantkit` 库、研究流程。方法论的“为什么”在 `HARNESS.md`,复盘结论在 `REVIEW_2026-10.md`,v5 仓位在 `V5_SIZING_EXPLAINED.md`。

## 1. 仓库地图

```
quantkit/                 可复用库(新工作从这里 import)
  data/    fetch.py       下载 Yahoo / Cboe / Ken French / SqueezeMetrics / moomoo 分钟线,原子写入
           load.py        统一读取(日线、VIX 族、French、DIX/GEX、经济日历、分钟 csv)
           bars.py        分钟线 → 日 × 分钟矩阵(适配 studies/etft)
           mdrec.py       读逐笔 + 10 档盘口记录
           manifest.py    MD5 清单写入与校验
           calendar.py    FOMC 决议日
  backtest/ daily.py      核心仓规则、分位数、趋势、不交易带、成本、波动目标
            portfolio.py  v5 组合层(含 PIT 买力缩放选项)
            intraday.py   噪声区间引擎适配(单相位 / 30 相位)
  stats/   perf.py        指标与分布面板
           bootstrap.py   配对块自助 ΔSharpe、平稳自助、max-t、Holm
           inference.py   HAC 均值检验、HAC 回归、聚类
           overfit.py     PSR / DSR / CSCV-PBO / 等波动增量
           diagnostics.py PSI、分位表与 MonoScore、IC、Barra-lite 归因
           cv.py          禁运、扩展窗口 + 内层留一年调参
  execution/ book.py      盘口逐档吃单、滑点表
  audit.py                投毒测试(前视检查)
examples/                 4 个可直接运行的例子
studies/                  190 个冻结的研究脚本(INDEX.md);etft/ 引擎;audit/ 审计与复盘脚本
docs/                     提案、报告、HARNESS、REVIEW、TRIAL_REGISTRY、本指南
outputs/{tables,figs}/    研究产物(已提交)
data/                     只放清单和说明;原始数据不入库
tools/                    服务器核对、实盘等价性检查
```

`studies/` 里的脚本互相按顶层模块名 import(`import tearsheet as TS`),**不要移动或改名**。它们是带预注册的研究记录,冻结。

## 2. 环境

```bash
cd IntradayMomentum
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .                     # 可选:让 `import quantkit` 在任何目录可用
export ETFT_DATA=/path/to/data       # 内含 raw/min1、raw/yahoo、raw/cboe ...;默认 <repo>/data
```

不装也能用:在仓库根目录运行,或 `sys.path.insert(0, "<repo>")`。`quantkit` 只在需要旧引擎时才把 `studies/` 加进 `sys.path`。

## 3. 数据

| 数据 | 路径 | 来源 | 怎么更新 |
|---|---|---|---|
| 日线(QQQ SPY MTUM ^NDX ^GSPC) | `raw/yahoo/*.parquet` | Yahoo(open/high/low 未复权,`close` = Adj Close,`close_raw` = Close) | `fetch.fetch_yahoo(t)` |
| VIX / VIX9D / VXN / VVIX | `raw/cboe/*.parquet` | Cboe 官方 `*_History.csv` | `fetch.fetch_cboe(s)` |
| 动量因子等 | `raw/french/*.csv` | Ken French 数据库 | `fetch.fetch_french(name)` |
| DIX / GEX | `raw/squeeze_dix_gex.parquet` | SqueezeMetrics | `fetch.fetch_squeezemetrics()` |
| 1 分钟 bar | `raw/min1/*.csv` | moomoo OpenD(服务器 `/opt/etf_timing_data/min1`) | `fetch.fetch_moomoo_min1`(**沙箱里没有 OpenD,未实测**)或从服务器拷贝 |
| 经济日历 | `raw/econ_cal/ec.csv` | 无公开源的复现脚本 | 沿用现有文件 |
| 逐笔 + 10 档盘口 | `mdrec/data/<日期>/` | 服务器上的 mdrec 记录器(`IntradayMomentumLive` 的 `feat/mdrec-recorder`) | 从服务器下载 |

一次性刷新所有有公开源的数据(约 5 秒):

```python
from quantkit.data import fetch
print(fetch.update_all())            # {'yahoo:QQQ': 'ok', ..., 'squeeze': 'ok'}
```

写入前会检查新数据行数不少于旧数据,否则拒绝覆盖(`force=True` 可强制)。

**已知问题**
- 沙箱里的 `raw/cboe/VIX.parquet` 与 Cboe 官方文件在 10 个日期上不同(最大差 2.61,例如 2026-02-06;最后一天 2026-10-20 是 19.05,官方收盘 17.31,存的是盘中值)。用 `fetch_cboe` 刷新后,依赖 VIX 的特征会有微小变化,历史结果不会逐位复现。需要复现旧数字时不要刷新,或先备份 `data/raw`。
- 本地日线和 VIX 到 2026-10-19/20,公开源已到 2026-11-05。分钟线和 mdrec 在服务器上,没有它们就不能做新的日内回测。
- 刷新数据后用 `manifest.write_manifest(path)` 写新清单,不要覆盖 `MANIFEST_sandbox_2026-10.md5`(它对应已发表的数字)。

## 4. 快速上手

```python
from quantkit.data import load
from quantkit.backtest import daily, portfolio
from quantkit.stats import perf, bootstrap, inference, overfit, diagnostics

px = load.load_close("QQQ"); r = px.pct_change().fillna(0.0)
core = daily.leg_return(daily.core_weight(px), r)             # 规则核心仓,2bp 单边成本
perf.compare({"B&H": r, "core": core}, start="2021")           # 分布面板
bootstrap.paired_delta_sharpe(core["2021":], r["2021":])        # 配对块自助的 ΔSharpe 区间
```

完整示例(都可以直接运行):

| 文件 | 内容 |
|---|---|
| `examples/01_data.py` | 加载、`--update` 刷新、清单校验 |
| `examples/02_core_backtest.py` | 核心仓 vs B&H:面板、分年、配对自助、HAC |
| `examples/03_stat_tests.py` | 候选族的增量 PSR/DSR、CSCV-PBO、PSI、分位单调性 |
| `examples/04_execution_cost.py` | TQQQ 盘口逐档吃单的滑点表 |

**分钟线 Sleeve**

```python
from quantkit.data.bars import load_bars
from quantkit.backtest import intraday
b = load_bars("QQQ")
ph = intraday.run_phases(b, offsets=range(30))                 # 每个相位约 3–5 秒
sleeve = intraday.phase_average(ph)                           # 回测用的 30 相位等权 Sleeve(PPP 之前)
```

这条路径与 `liquidity_inverse_study.run_phases_fills` 的缓存逐位一致(相位 0 已核对)。要得到 v4 的 Sleeve(入场带、FFA-P 跳过、PPP 的 k),用 `studies/r3_study.py` 的流水线;`studies/tearsheet_v5.py::main` 是最完整的接线示例。

**v5 组合层**

```python
res = portfolio.build_v5(core_w, sleeve_v4, e_vt, k_max, qqq_ret, ref_port=v4_port, rho=1.5, pit_cap=True)
res["port"], res["core_w"], res["sleeve_mult"]
```

输入的含义见 `portfolio.py` 文件头和 `V5_SIZING_EXPLAINED.md`。

## 5. 与旧脚本的等价性(已核对,2026-10)

| quantkit | 旧实现 | 结果 |
|---|---|---|
| `daily.core_weight` | `noise_core_dd.core_weights(px)[0]['base']` | 最大差 0.0 |
| `perf.distribution_panel` | `sizing_sigma_feed.panel` | 13 项全部 0.0 |
| `bootstrap.fixed_block_indices` | `sizing_models_study.block_idx` | 下标逐位相同 |
| `overfit.psr / expected_max_sr / cscv_pbo` | `audit/overfit_audit.py` | PSR、PBO 相同,expected_max_sr 差 2e-15 |
| `portfolio.build_v5`(pit_cap=False / True) | `tearsheet_v5.build_v5`(kmax 实值 / 常数 2) | 最大差 0.0 |
| `execution.book.slippage_table` | `hf_exec_cost.py` | 表中数字相同 |
| `intraday.run_phases(offsets=[0])` | `liquidity_inverse_study.run_phases_fills()[0]` | 日收益差 0.0,成交行数相同 |
| `data.calendar.FOMC_DECISION` | `studies/event_calendar.py` | 相同 |

## 6. 做一项新研究的流程(摘要,细节见 HARNESS.md)

1. **预注册**:在脚本头(或 `docs/PROPOSAL_*.md`)写假设、配置清单、判据、数据门槛。先提交,再运行。
2. **数据门槛**:校验分钟线 MD5,复现基线的关键数字。
3. **特征先过投毒测试**:`quantkit.audit.poison_future / poison_intraday`。
4. **比较**:候选和基线用同一组块自助下标(`paired_delta_sharpe`);同时看过多个候选时用 `max_t_pvalues` 和 `holm`。
5. **报告**:分布面板,不只 Sharpe;分年;去掉整个新增层的对照。
6. **多重检验**:在 `docs/TRIAL_REGISTRY.md` 追加一行并更新“合计”;对**增量**算 DSR,不对基础策略算。
7. **证据等级**:E0–E3 见 HARNESS.md;“test 2023–26”已不是干净留出集,前瞻从 2026-11-02 开始。
8. **成本**:订单规模超过 ≈2k 股 TQQQ 时,先用 `execution.book` 重算滑点。

最小脚本模板:

```python
"""<研究名> —— 预注册(运行前写定,YYYY-MM-DD)
假设: ...
配置(共 N 个,不增删): ...
判据: ΔSR 的配对块自助 95% 下界 > −0.05,且 ...
数据门槛: ...
"""
import sys; sys.path.insert(0, "<repo>")
from quantkit.data import load
from quantkit.stats import perf, bootstrap
# 1) 构造候选与基线  2) 判据  3) 打印分布面板  4) 写 outputs/tables/<name>.csv
```

## 7. 常见坑(项目里真实踩过的)

| 坑 | 防法 |
|---|---|
| 日频特征用了当日信息 | 特征 `shift(1)`;投毒测试 |
| 把日内才知道的量(如 kmax)用在整日收益上 | 只用开盘前可知的量;PR #2 在 v5 里发现过 |
| 把 MOC 出场换成 15:40 主动出场却不加半价差 | 成本跟着执行方式走 |
| DSR/PBO 作用在基础策略或只变化风险偏好的候选族上 | 对增量算;候选族要有实质差异 |
| 用同一段“测试集”做了几百次决策 | 登记簿 + 前瞻影子盘 |
| 内层交叉验证没有隔离相邻日 | `embargo`(默认 0 保持旧结果;新研究用 21) |
| 订单变大后仍用 1 tick 半价差 | `execution.book` |
| 刷新数据后旧数字不复现 | 刷新前备份;写新清单 |
| 把 Sharpe 当唯一指标 | 分布面板 |
