# PROPOSAL — Noise Band 的统计含义、校准与策略增量

状态：**历史诊断方案已冻结；外部 SPY 2021–2024 已运行，canonical QQQ/v5 待数据**
分支：`research/noise-band-validation-2026-10`
基准提交：`origin/research/crowding-vol-timing-2026-10@8a0f5b51c6cd68b613c713e109d339033978c3b4`
冻结日：2026-10-31；冻结历史截至 **2026-10-30**，前向确认从冻结后的首个交易日 **2026-11-02** 起计。至少积累 252 个交易日后才作确认性判断。

## 1. 先给结论

现有代码实现了论文规则，但“Noise Band 捕捉了 noise range”还不是一个经过统计检验的结论。

原因是原始宽度

\[
\hat m_{t,k}=\frac{1}{14}\sum_{j=t-14}^{t-1}|C_{j,k}/O_j-1|
\]

估计的是同一时点**绝对位移的条件均值**，不是一个带有明确覆盖概率的预测区间。`vm=1` 没有 68%、80% 或 95% 等统计含义。即使条件分布是零均值正态，`E|X|` 所覆盖的中心概率也只有约 57.5%，真实日内收益的厚尾、偏度、隔夜缺口和波动聚集还会改变这个数字。

当前 v4/v5 使用的 `s3`：

\[
s_{3,t,k}=\operatorname{clip}\left(\sqrt{\frac{RV_{t,k}/S^{RV}_{t,k}}{V_{t,k}/S^V_{t,k}}},1,2\right)
\]

只放宽入场带，不改变止损带。它有合理的研究动机：当实现波动上升却没有累计成交量配合时，把该价格变化视作更可能是噪声。但它仍有三个未经证明的前提：

1. 1 分钟总成交量可以代表信息到达或主动订单流。总量没有买卖方向，也没有盘口信息。
2. `RV surprise / volume surprise` 的平方根是正确函数形式。
3. 乘数下限为 1、上限为 2 的单边放宽优于重新校准整个条件分布。

仓库现有结果对 `s3` 只提供弱支持：

- `outputs/tables/r2_rules.csv`：C3 相对 v3 的组合 ΔSharpe 为 **+0.039**，95% 区间 **[-0.138, +0.201]**。
- `outputs/tables/r3_configs.csv`：只改入场带的 E1 ΔSharpe 为 **+0.039**，95% 区间 **[-0.138, +0.201]**；Sleeve 最差日从约 **-250bp** 恶化到 **-332bp**。
- `outputs/tables/r3_ladder.csv`：E1 之后加入 `rv_surp` 替代特征反而贡献 **-0.073 ΔSharpe**；后续改善主要来自 `dur_rate`，不是 Noise Band 本身。
- `outputs/tables/tearsheet_v5_steps.csv`：v5 的高 Sharpe 主要是买力上限、核心仓不交易带和 Sleeve/core 配比共同作用后的组合结果，不能反推 `s3` 已经被验证。

因此当前判断是：**原始 band 有可解释的季节基准作用，s3 有方向一致但不显著的历史线索；两者都还不能被称为统计上已校准的 noise range。**

## 2. 已检查的代码与版本范围

本方案建立前检查了：

- `main` 和五个远程 `research/*` 分支；
- 唯一 PR #1。它仍为 open，head 是 `research/layered-2026-10@3f5902b`，没有 review、issue comment 或 inline comment；
- 最新研究链：`main → layered → duration-vol-entry → intraday-vol-entry → crowding-vol-timing`；
- 日内 band 的最新实质变化是 `intraday-vol-entry` 的 E1。最新 `crowding-vol-timing` 分支没有再改日内入场带，只增加了核心仓 no-trade band 与 v5 组合工程；
- [QuantMLResearch/quant-handbook](https://github.com/QuantMLResearch/quant-handbook) 全仓。可借鉴的是 HAC、purged forward split 和 DSR 的研究纪律；该仓库没有 Noise Band、block bootstrap、White Reality Check、SPA 或完整 PBO 实现。

## 3. 研究问题与事前假设

### H1：band 是否是稳定的 noise range

分别检验：

- **raw coverage**：`|C_k/O-1| <= width_k`；
- **anchored coverage**：价格是否位于使用 `max(open, prev_close)` 与 `min(open, prev_close)` 后的实际上下轨内；
- 覆盖率是否随年份和时点稳定；
- 固定 60% 分位带是否在未参与估计的当日接近 60% raw coverage。

原始 mean14 和当前 s3 没有名义覆盖率，因此不对它们做虚构的 Kupiec 式“正确覆盖率”检验，只报告覆盖率、年份漂移和时点漂移。q60 带有 60% 名义 raw coverage，使用日级 coverage 序列的 HAC 检验，避免把同一天多个时点当作独立观测。`s3` 使用截至当前 bar 收盘的 realized variance 与成交量，所以它的 coverage 是**同时点描述性 coverage**，不是在该 bar 开始前形成的 interval forecast；机制判断主要依赖后续收益，而不是把 s3 coverage 当预测校准。

### H2：出界是否识别了“非噪声”

在 10:00、10:30、…、15:00 的决策点，按下一分钟开盘成交，测量 5 分钟、30 分钟和 MOC 的方向收益。

主统计量：出界事件的方向延续收益。
增量回归：

\[
y_{t,k,h}=a+b\,1(\text{outside})+c\log(1+distance)+d\log(s3)+e|gap|+\text{clock FE}+u_{t,k,h}
\]

标准误按交易日聚类。`b` 回答的是：控制连续位移大小、当日波动/成交量状态、缺口和时点后，band 的离散阈值是否仍增加预测力。若 `b` 不稳定或不显著，band 仍可作为交易规则，但“它分隔 noise 与 information”的机制说法不成立。

### H3：能否提高当前 v5 的净策略表现

只比较四个事前固定配置：

| ID | 入场宽度 | 用途 |
|---|---|---|
| `paper_mean14` | 14 日同分钟绝对位移均值 | 原论文对照 |
| `current_mean14_s3` | `mean14 × s3` | 当前 v4/v5 基准 |
| `q60_l60` | 过去 60 日同分钟绝对位移的 60% 分位 | 有明确覆盖含义的候选 |
| `q60_l60_s3` | `q60_l60 × s3` | 检验成交量/波动状态在校准带上是否仍有增量 |

60% 是运行前固定值，不扫描。它接近正态假设下 `E|X|` 隐含的 57.5% 覆盖率，同时便于解释。60 日窗口减少 14 个观测上估计分位数的离散性。任何其他分位数、窗口、上下侧不对称或函数形式都属于下一轮研究，不能追加到本轮后挑胜者。

## 4. 策略实验如何隔离 band

`studies/noise_band_validation.py` 对每个候选：

1. 只替换**入场**宽度；止损、hold/rearm 和 tail 规则仍用原 band；
2. 四个候选全部在同一进程重跑 30 个 phase 和 FOMC late-entry 路径，不读取无 provenance 的旧 phase cache，也不对旧成交做事后过滤；
3. 保留 FFA 硬跳过、周一不交易和 15:40 平仓；对 15:40 仍持仓的交易改用 15:41 open，并补收原 v4/v5 P&L subtraction 漏掉的主动退出 half-spread；
4. 在新的腿收益上按现行 expanding walk-forward 协议重训 `CLEAN + dur_rate` 线性仓位模型；年份循环随数据动态扩展，缺少未来 FOMC 日历时 fail closed；
5. 使用 v5 的固定 `CAP=2.57`、核心仓 no-trade band `0.10` 和 `rho=1.5`；cap 在开盘前按仓位模型已知上界 `k=2` 计算，不用当日 10:00 才产生的 `k` 回写开盘以来核心仓收益；因此输出是 **PIT-corrected v5 comparison**，不会机械复现仓库原 v5 数字；
6. 所有候选用当前基准 pre-v5 组合的滞后 252 日波动作为共同风险目标，避免候选靠增加风险提高均值；
7. 报最终组合与 Sleeve 的净结果，同时分开报告双边 TQQQ 名义换手和单边核心仓换手、入场次数、持仓时间、CVaR 和最差日。

## 5. 推断与多重检验

主要推断单位是**每日最终组合净收益**，不是逐分钟或逐笔事件。

- 候选减当前基准的每日净收益：Newey-West HAC，lag=10；
- ΔSharpe 和年化均值差：同步、循环 stationary bootstrap；平均区块长度 5、10、20 日；
- 三个候选共同检验：Holm 校正 HAC 单边 p 值，并报告基于同步 block resample 的 single-step max-t family-wise p 值；
- 5/30 分钟事件重叠：回归标准误按交易日聚类；
- dev/test：仍输出冻结的 2019–2022 与 2023–2026-10-30，但明确标成 `legacy_*`，不再称为真正未触碰样本外；
- forward 少于 252 个交易日时标记 `confirmatory_ready=False`，不输出可用于采用决策的 Holm/max-t p 值；达到门槛后，对冻结候选族单独计算 forward Holm 与同步 max-t。

本实验不报告“校正后已无过拟合”的结论。仓库已在相同历史上检查超过 300 个配置，且没有完整的历史 trial ledger；DSR、PBO、Reality Check 都无法逆向消除这条研究路径。历史检验只能衡量估计不确定性。

## 6. 采用门槛

### 对当前设计的机制判断

必须同时满足：

1. anchored coverage 的跨年和跨时点漂移可接受；
2. 出界后的 30 分钟或 MOC 延续收益为正，且交易日聚类 t 值在 dev/test 同号；
3. 控制连续位移、s3、gap 和 clock 后，`outside` 系数仍为正；
4. QQQ 与 SPY 的方向一致。

否则只把 band 称作“季节调整的突破阈值”，不称作经过验证的 noise range。

### 对策略替换的判断

新候选替换 `current_mean14_s3` 需要：

1. 前向确认样本至少 252 个交易日；
2. 风险匹配后的年化净均值差为正，max-t family-wise 单边 p ≤ 0.05；
3. block 5/10/20 的 ΔSharpe 95% 区间下界均不低于 -0.05；
4. dev/test/forward 方向一致；
5. CVaR5%、最差日和最大回撤不恶化超过 10%；
6. TQQQ 名义换手不增加超过 10%，除非净成本后收益增量已经覆盖。

历史样本即使全部满足，也只能产生“前向候选”，不能直接升级 v5。

## 7. 数据门槛与输出

默认验证 `data/MANIFEST_sandbox_2026-10.md5` 的全部依赖；纯 calibration 模式只要求并校验指定标的分钟线。原始数据不在 Git 仓库内，需设置：

```bash
export ETFT_DATA=/path/to/data
python studies/noise_band_validation.py --section all --bootstrap 2000
```

输出：

- `outputs/tables/noise_band_coverage.csv`
- `outputs/tables/noise_band_coverage_by_year.csv`
- `outputs/tables/noise_band_coverage_by_clock.csv`
- `outputs/tables/noise_band_continuation.csv`
- `outputs/tables/noise_band_incremental_regression.csv`
- `outputs/tables/noise_band_strategy.csv`
- `outputs/tables/noise_band_inference.csv`
- `outputs/tables/noise_band_strategy_daily.csv.gz`
- `outputs/tables/noise_band_standalone_strategy.csv`
- `outputs/tables/noise_band_standalone_inference.csv`
- `outputs/tables/noise_band_experiment_metadata.json`
- `outputs/tables/noise_band_experiment_metadata_<section>.json`

## 8. 文献依据

- 原始策略：[Zarattini, Aziz & Barbon, *An Effective Intraday Momentum Strategy*](https://www.alexandria.unisg.ch/bitstreams/a99aba00-f967-49b3-aceb-f544dc386e0b/download)。
- 日内动量与高波动/高成交量状态：[Gao, Han, Li & Zhou, *Market Intraday Momentum*](https://c.mql5.com/forextsd/forum/173/intraday_momentum_-_the_first_half-hour_return_predicts_the_last_half-hour_return.pdf)。
- 日内周期性：[Andersen & Bollerslev, *Intraday Periodicity and Volatility Persistence in Financial Markets*](https://www.kellogg.northwestern.edu/academics-research/research/detail/1997/intraday-periodicity-and-volatility-persistence-in-financial-markets/)。
- 成交量作为随机时钟/信息流代理的起点：[Clark, *A Subordinated Stochastic Process Model*](https://conservancy.umn.edu/items/09b64a82-e67d-4073-a494-6f9ad47350c0)；[Tauchen & Pitts, *The Price Variability-Volume Relationship*](https://public.econ.duke.edu/~get/wpapers/TP1983.pdf)。这些文献提供动机，不证明本项目的 `s3` 公式。
- 区间预测应同时检查覆盖率与例外聚集：[Christoffersen, *Evaluating Interval Forecasts*](https://www.researchgate.net/publication/2432264_Evaluating_Interval_Forecasts)。本实验因日内观测相关，采用日级 HAC/聚类，而不直接套独立 Bernoulli 假设。
- 依赖时间序列重采样：[Politis & Romano, *The Stationary Bootstrap*](https://www.math.ucsd.edu/~politis/SBblock-revER.pdf)。
- 数据窥探和模型族联合检验：[White, *A Reality Check for Data Snooping*](https://agnostic.encelade.ch/intro/history/White2000.pdf)；[Bailey et al., *The Probability of Backtest Overfitting*](https://escholarship.org/content/qt4w1110bb/qt4w1110bb.pdf)；[Bailey & López de Prado, *The Deflated Sharpe Ratio*](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)。

以上网络资料均作了转述，没有复制受限原文。Content was rephrased for compliance with licensing restrictions.
