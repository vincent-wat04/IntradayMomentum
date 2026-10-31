# Noise Band 外部 SPY 回测：2021–2024

日期：2026-10-31
性质：非 canonical、历史诊断；不能替代 QQQ/v5 主回测或冻结后的前向验证。

## 数据

在当前沙盒没有 `ETFT_DATA`、moomoo OpenD、行情 API key、SSH 数据主机或 GitHub artifact 的情况下，使用公开 GitHub 仓库 [Branly76/Intraday-strategy-Beat-the-market-for-SPY-](https://github.com/Branly76/Intraday-strategy-Beat-the-market-for-SPY-) 中随项目提供的 Alpaca SPY 1 分钟文件作外部诊断。

- 外部仓库 commit：`90c392dd2374dde0e1cec9aeb8654a7864de483e`
- 原始文件 SHA-256：`ab9b2790a7af73a16d9776ad81eefdbd7eb4f456e4b4419a95cd1ba0f0bed8eb`
- 原始记录：391,195 行，2021-01-04 09:30 至 2024-12-30 15:59，美东时间；
- Alpaca bar 起始标签统一加一分钟，转换为本项目的 bar 结束标签；
- 经 `etft.intraday.bars.load_bars` 后保留 995 个完整交易日，每日均为 390 根 bar；策略比较再统一剔除 q60 的 42 日预热期，四个版本使用相同的 953 日；
- 用 yfinance `auto_adjust=False` 日线建立 `close/close_raw`，只用于除息日前收调整；
- 外部仓库没有明确开源数据许可证，README 也没有授予再分发权。因此原始数据只留在被 `.gitignore` 排除的 `data_external/`，PR 只提交聚合统计，不提交或复制原始行情。

这套数据不是仓库 MD5 manifest 中的 moomoo canonical 数据，时间也只覆盖 2021–2024。结果用于回答“候选是否值得继续”，不能用于声称已复现仓库 QQQ v5。

## 方法

运行命令（两步分别生成 calibration/机制表和 standalone 策略表）：

```bash
ETFT_DATA=/projects/sandbox/IntradayMomentum/data_external \
python studies/noise_band_validation.py \
  --section calibration --symbols SPY --skip-md5 --bootstrap 2000 \
  --data-label public-alpaca-spy-2021-01-04_to_2024-12-30@90c392dd

ETFT_DATA=/projects/sandbox/IntradayMomentum/data_external \
python studies/noise_band_validation.py \
  --section standalone --symbols SPY --skip-md5 --bootstrap 2000 \
  --data-label public-alpaca-spy-2021-01-04_to_2024-12-30@90c392dd
```

Standalone 模式只隔离 Noise Band，不加载 QQQ 核心仓、FFA、PPP 或 v5 配比：

- 当前审计状态机；
- 30 个检查相位等权；
- `band+tail`、`tail_k=0.5`、逐 bar tail stop；
- `reentry=hold`；
- 2% 日波动目标，最大 4x；
- 周一不交易；
- 决策 bar 收盘观察，下一 bar 开盘成交；
- 每股半价差 $0.005、佣金/滑点 $0.0045，MOC 仅收佣金；退出成本使用入场时实际股数，而不是旧引擎的 `leg_price` 近似；
- 三个候选相对 `current_mean14_s3` 做 HAC、Holm、同步 max-t 和 stationary bootstrap，平均区块长度 5/10/20 日。

## 1. 是否真的形成 noise range

2021–2024 canonical 检查点结果：

| Band | raw coverage | gap-anchored coverage | 平均全宽 |
|---|---:|---:|---:|
| paper mean14 | 58.38% | 70.83% | 126.7bp |
| current mean14 × s3 | 60.88% | 72.47% | 133.2bp |
| q60-L60 | **60.90%** | 72.35% | 128.1bp |
| q60-L60 × s3 | 63.18% | 73.94% | 134.4bp |

q60-L60 的目标是 60%。实际 raw coverage 为 60.90%，日级 HAC 检验 `coverage−60%=0.90%`，`t=0.54, p=0.589`。因此本样本不能拒绝 60% 校准。

这个结果支持一个狭义结论：**q60-L60 比 mean14 更适合被称为具有明确统计含义的 noise range。** current mean14×s3 恰好也得到约 60.9% coverage，但 s3 使用当前 bar 的 realized variance 和 volume，只能称为同时点描述性 coverage，不能称为 bar 前预测区间。

## 2. 带外是否意味着后续延续

全部样本中，带外事件的 30 分钟方向收益均为正：

| Band | 事件数 | 30 分钟均值 | 按日聚类 t |
|---|---:|---:|---:|
| paper mean14 | 3,164 | +1.22bp | 3.33 |
| current mean14 × s3 | 2,986 | +1.22bp | 3.43 |
| q60-L60 | 2,899 | +1.01bp | 2.49 |
| q60-L60 × s3 | 2,732 | +1.16bp | 2.86 |

但在 2023–2024 段，控制连续位移、s3、隔夜缺口和时点固定效应后，`outside` 阈值的增量系数都不显著：

| Band | 30 分钟 outside beta | t | p |
|---|---:|---:|---:|
| current mean14 × s3 | +0.43bp | 0.60 | 0.549 |
| q60-L60 | +0.21bp | 0.27 | 0.790 |
| q60-L60 × s3 | +0.34bp | 0.42 | 0.676 |

因此数据支持“较大的同方向开盘位移后，平均存在短期延续”，但不支持更强的机制说法，即某条离散 band 能稳定地把 noise 和 information 分开。连续位移强度可能解释大部分预测力。

## 3. 策略表现

### 全部 2021–2024

| Band | 日均 | Sharpe | CAGR | 波动 | MDD | 双边日换手 |
|---|---:|---:|---:|---:|---:|---:|
| paper mean14 | +7.46bp | 1.425 | 19.65% | 13.20% | -7.62% | 2.75x |
| **current mean14 × s3** | **+8.05bp** | **1.614** | **21.53%** | 12.56% | -6.25% | 2.58x |
| q60-L60 | +6.51bp | 1.275 | 16.85% | 12.85% | -7.70% | 2.51x |
| q60-L60 × s3 | +7.12bp | 1.468 | 18.78% | **12.23%** | **-6.01%** | **2.33x** |

相对当前 mean14×s3：

- paper mean14：ΔSharpe `-0.189`，block-10 区间 `[-0.390,+0.010]`；
- q60-L60：ΔSharpe `-0.339`，区间 `[-0.754,+0.061]`；
- q60-L60×s3：ΔSharpe `-0.146`，区间 `[-0.535,+0.232]`；
- 三个候选的 Holm p 均为 1，block-10 max-t p 分别为 0.980、0.996、0.972。

### 2023–2024 段

| Band | Sharpe | MDD |
|---|---:|---:|
| current mean14 × s3 | 1.497 | -6.01% |
| q60-L60 | 1.521 | -6.87% |
| q60-L60 × s3 | 1.520 | -6.01% |

q60 两个版本的点估计略高约 `+0.024 ΔSharpe`，但 block-10 区间分别为 `[-0.374,+0.399]` 和 `[-0.330,+0.397]`，HAC 单边 p 为 0.511 和 0.579。两者都没有统计增量。

q60-L60×s3 相比当前 band 将双边日换手从 2.58x 降至 2.33x，约降低 9.6%，MDD也略低；代价是全样本均值和 Sharpe下降。它可以保留为低换手 challenger，但不能替换当前设计。

## 结论

本次真实 SPY 外部数据回测否定了“q60 已经比当前 band 更赚钱”的假设：

1. **统计定义**：q60-L60 明显更可解释，并在本样本实现了目标 60% coverage；
2. **机制**：所有 band 的带外事件都有约 1bp 的 30 分钟延续，但阈值在后半样本没有超出连续位移大小的显著增量；
3. **策略表现**：当前 mean14×s3 在 2021–2024 全样本仍是最高 Sharpe；
4. **改进候选**：q60-L60×s3 降低约 9.6% 换手和部分尾部风险，但收益下降，统计上没有优于当前版本；
5. **决策**：保留当前 Noise Band，不升级 q60。下一步应使用 canonical QQQ 数据运行 PIT-corrected v5，并把 q60×s3 只作为预先冻结的低换手 challenger。

聚合结果：

- `outputs/tables/noise_band_coverage.csv`
- `outputs/tables/noise_band_continuation.csv`
- `outputs/tables/noise_band_incremental_regression.csv`
- `outputs/tables/noise_band_standalone_strategy.csv`
- `outputs/tables/noise_band_standalone_inference.csv`

外部资料内容均作了转述。Content was rephrased for compliance with licensing restrictions.
