# 原始时序因子在择时中如何处理：文献 + 实证（SPY/QQQ，2019-01 → 2026-10）

脚本：
- `timing_transform_study.py`：单因子 × 时间聚合 × 映射。
- `placebo_transform.py`：零假设基准。
- `timing_combination_study.py`：多因子合成。
- `timing_transform_report.py`：出图和汇总表。

结果在 `outputs/full/transform/`。dev 为 2019–22，test 为 2023-01-03 之后。所有统计量都只用信号日收盘前可得的数据（PIT）；所有需要估计的量，都只用到在信号日收盘前已经到期的标签。执行方式为 next_open，成本 1bp。

## 1. 之前的"单因子 Sharpe"是怎么算的（`factor_sharpe.py`）
1. **输入**：仓库正交化后的因子列，即因子的 K 日均值减去它对过去 K 日收益做 120 日滚动回归后的残差。宏观因子也做了同样的处理。
2. **信号**：s = 2×（因子在过去 252 日内的分位数）− 1，取值在 [−1, 1]。
3. **方向**：用 2019–22 已到期标签的 Pearson IC 符号，确定后冻结。
4. **平滑**：做 h 日移动平均，相当于 h 个重叠子组合。
5. **回测**：
   - 决策在 t 日收盘，t+1 开盘调仓；
   - 隔夜仍按旧仓位持有；
   - LS 仓位为 w，LC 仓位为 1{w>0}，TILT 仓位为 0.5+0.5w；
   - 日收益年化 Sharpe = 均值/标准差×√252。
6. **dev 期 Sharpe 是样本内的**，因为方向是在这一段上定的。

**问题**：
- 这只是一种处理方式。
- 正交化会去掉因子中"过去收益"的成分。噪声带类因子与过去收益的相关高达 0.64–0.81，正交化之后几乎没有剩下东西。
- 没有零假设基准。

## 2. 文献里原始预测变量的处理方式
| 环节 | 做法 | 代表文献 |
|---|---|---|
| 标准化 | 滚动或扩张窗口的 z 分数、分位数；用截尾/分位数区间压缩极值，例如 100% + (EP−中位数)/(95–5 分位区间)，下限 50%，上限 150% | [Asness, Ilmanen & Maloney 2017 "Sin a little"](https://www.aqr.com/Insights/Research/Journal-Article/Market-Timing-Sin-a-Little) |
| 离散化 | 移动平均、动量、OBV 等 0/1 信号，再做主成分或投票 | [Neely, Rapach, Tu & Zhou 2014](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2013.1838) |
| 显著性门槛 | 只在趋势统计显著时持仓，可以大幅降低换手，业绩不受显著影响 | [Baltas & Kosowski](https://spiral.imperial.ac.uk/server/api/core/bitstreams/93f06c22-150a-4240-8115-912988dd2c67/content) |
| 预测回归 | 扩张窗口 OLS 得到 μ̂；对系数和预测值施加符号约束，可以小幅但有经济意义地打败历史均值 | [Campbell & Thompson 2008](https://dash.harvard.edu/handle/1/2622619)；[Goyal & Welch 2008](https://www.nber.org/papers/w10483)；[Goyal, Welch & Zafirov 2024](https://sfs.org/rfs-forthcoming-paper-9/) |
| 预测组合 | 单变量预测取均值或中位数，可以显著降低预测方差 | [Rapach, Strauss & Zhou 2010](https://www.jstor.org/stable/40468327) |
| 降维、监督合成 | PCA；PLS / 3PRF | [Kelly & Pruitt 3PRF](https://www.researchgate.net/publication/228206332_The_Three-Pass_Regression_Filter_A_New_Approach_to_Forecasting_Using_Many_Predictors)；[Huang, Jiang, Tu & Zhou 2015](https://academic.oup.com/rfs/article-abstract/28/3/791/1576380) |
| 稀疏组合 | C-ENet：用 elastic net 先挑出单变量预测，再取平均 | [Dong, Li, Rapach & Zhou 2022](https://www.researchgate.net/publication/356805694_Anomalies_and_the_Expected_Market_Return)；[综述](https://oxfordre.com/economics/view/10.1093/acrefore/9780190625979.001.0001/acrefore-9780190625979-e-777?rskey=jFxXeb) |
| 波动率 | 统计上显著的预测变量，如果在高波动期取极值、持续性低或尾部厚，经济价值会很小；仓位应按 μ/(γσ²) 设定 | [Cederburg, Johnson & O'Doherty 2023](https://experts.arizona.edu/en/publications/on-the-economic-significance-of-stock-return-predictability)；[Moreira & Muir](https://www.nber.org/papers/w22208)；[Cederburg et al. 2020 的样本外质疑](https://experts.arizona.edu/en/publications/on-the-performance-of-volatility-managed-portfolios/) |
| 日内信息 | 开盘半小时收益预测收盘半小时收益，信息在日内就被消化掉了 | [Gao, Han, Li & Zhou 2018](https://scispace.com/papers/market-intraday-momentum-1signxa1z5) |

（以上为文献内容的转述。）

## 3. 实验设计
- **因子**：128 个原始日频因子。
  - 21 个日内/量价因子；
  - 5 个噪声带因子：MID_MED（中位数）、MID_AVG、MID_END、OUT_AVG、SKEW_DIFF（均值−中位数，即偏度代理）；
  - 102 个宏观因子。
- **时间聚合（7 种）**：L1（当日原值）、MA5、MA20、EWM10（半衰期 10）、MED20（20 日中位数）、DEV60（与 60 日均值的偏离）、CHG20（变化量）。
- **映射（15 种）**：
  - 滚动 252 日分位数 PCT252，扩张分位数 PCTEXP；
  - 滚动 z 分数 Z252，扩张 z 分数 ZEXP；
  - 稳健 z 分数 RZ252（中位数/MAD）；
  - 符号 SIGN，尾部阈值 TAIL（分位 >80% 或 <20% 才持仓）；
  - 波动缩放 VOLZ；
  - 方向的三种定法：dev 冻结、扩张 OLS（Z252_xs）、扩张秩 IC（Z252_xr），以及经济先验方向（Z252_prior）；
  - 预测回归：REG、REGCT（CT 约束）、REGV（目标除以波动率）、REGR（秩 IC 加权）。
- **零假设**：
  - 把每个真实因子循环平移至少 1 年，各做 2 次。这样保留了因子的分布、持续性和缺失结构，但切断了它与未来收益的联系。
  - 另外加入 60 个白噪声 / AR(1) 序列。
- **组合**：9 种合成方法 × 3 个因子池（INTRA / MACRO / ALL）。每个组合把整块因子面板一起平移 4 次作为零假设。

## 4. 结果

### 4.1 因子有信息，但很弱
- **h=20：** 每一种 z 分数或分位数映射，测试期 IC 都比零假设高约 0.025–0.03，按因子聚类的 t 值为 3.0–3.9。h=1 时 t 约为 3.5，h=5 时约为 2。
- **换成 Sharpe：** 单因子纯择时多空的超额 Sharpe 只有 +0.03 到 +0.09。3.75 年的测试期里，Sharpe 的标准误约 0.5，所以单个因子不可能显著。之前算出来的中位 Sharpe 约 0.1，原因是信号本身弱，不是计算有错。

### 4.2 映射方式（图 `fig1_maps.png`，表 `summary_maps.csv`）
- **滚动窗口优于扩张窗口。** RZ252 和 Z252 最好（h20 测试 Sharpe 0.058 / 0.037，零假设 0.012 / 0.003）。PCTEXP 和 ZEXP 的 Sharpe 低于零假设，但 IC 并不低：扩张窗口的统计量会被因子水平的长期漂移带偏，仓位长期偏向一边。这和 AQR 讲的估值漂移问题是同一个现象。
- **稳健 z 分数（中位数/MAD）在三个期限上都是第一。**
- **SIGN、TAIL 略差于连续映射。** 尾部阈值只有在信号足够强时才有价值。
- **回归类映射（REG*）的 Sharpe 全部为 −0.14 到 −0.25，这是机制性偏差，不是负信息。** 白噪声加上 iid 收益的模拟就能复现：β_t·x_t 的 Sharpe 均值约为 −0.15（t ≈ 13），而 sign(β)·x 约为 0。
  - **成因：** 递归估计的斜率会形成"下注规模反馈"。赢了之后 |β| 变大，输了之后 |β| 变小，于是每条路径的收益均值与波动率相关，Sharpe 被系统性压低。
  - **扣除零假设之后：** REGR（扩张秩 IC × z）在 h20 的超额 Sharpe 为 +0.08（t = 2.7），是所有映射中最高的；OLS 类的 REG 和 REGV 没有超额。
  - **结论：** 扩张窗口 OLS 被 2020 和 2022 年的极端样本主导，应该改用秩 IC 或稳健估计。凡是带递归估计的策略，Sharpe 都必须对照零假设来解读。
- **方向的定法：**
  - 冻结的 dev 方向和扩张秩 IC 方向（Z252_xr，h20 超额 +0.068）都可以用；
  - 扩张 OLS 方向（xs）没有超额；
  - 经济先验方向（registry 里的 prior）只有约 50% 和数据一致，用它来定方向反而亏损（h20 −0.075）。原因是恐慌簇因子的真实方向是逆向的（恐慌越高、后续收益越高），先验却按"风险越高越看空"来定。

### 4.3 时间聚合（`fig2_aggregation.png`，`summary_aggs.csv`）
- **聚合窗口要和预测期限匹配。**
  - h=20：EWM10、MA20、MED20 最好，IC 超额约 0.036–0.039（t ≈ 3.4–3.7）。
  - h=5：L1 和 MA5 最好，超额 Sharpe +0.07（t ≈ 2.1）。
- **变化量 CHG20 没有任何信息。** 去趋势的 DEV60 介于两者之间。

### 4.4 噪声带"中位数 / 偏度"因子（`fig3_noise_band_factors.png`，`summary_noise_factors.csv`）
| 因子 | 结论 |
|---|---|
| MID_MED / MID_AVG / MID_END（价格在噪声带中的位置，中位数/均值/尾盘值） | 方向稳定为负（逆向）。h=1 时测试 Sharpe 0.28–0.30，80–93% 的变体为正；h20 时 0.10–0.19，IC ≤ 0.04。但它们与过去 K 日收益的相关为 0.64–0.81，正交化后 IC ≈ 0（`base_results.json`）。**实质上是短期反转因子的另一种写法，没有独立信息。** |
| SKEW_DIFF（均值−中位数，偏度代理） | **无效**。不同标的和聚合方式下 dev 方向不一致；测试期 Sharpe 为负（h1 −0.16，h20 −0.22），h20 IC −0.05，方向翻转。 |
| VP_MED_SKEW / LOGVOL_SKEW（量价分布偏度、成交量偏度） | 无效，方向不稳定。 |
| OUT_AVG（带外超额距离） | 很弱。 |
| 对照组 VP_REVERSAL | 唯一明显有效的日内因子：测试 Sharpe 0.39–0.61，89–99% 的变体为正，零假设约 0.03。 |

解释：噪声带的信息属于日内动量，在当天就被策略本身消化了（参见 Gao et al. 2018，以及我们的 E19–E29）。折算到日频以上的期限后，剩下的只有反转成分。

### 4.5 多因子合成（`fig4_combinations.png`，`fig5_equity.png`，`summary_combinations.csv`）
**纯择时 LS，h20，测试期 Sharpe（零假设均值约 −0.6 到 +0.4，标准差约 0.5）：**

| | 简单等权 EW_DEV / 投票 VOTE | 预测类（RSZ / PCA3 / PLS1 / C-LASSO） |
|---|---|---|
| SPY INTRA | 0.40 / 0.47 | 最好为 C-LASSO 0.32（R²_OS 3.1%） |
| QQQ INTRA | 0.43 / 0.53 | — |
| MACRO | 0.21–0.27 | — |
| ALL | 0.25–0.37 | — |
| 测试 IC | 0.10–0.27 | — |

- **预测类方法普遍更差。**
  - PCA3 在 QQQ h20 上为 −1.2；
  - PLS1 的 R²_OS 为 −10% 到 −30%（过拟合了水平项）；
  - RSZ 均值组合会收缩回历史均值，这与 RSZ 2010 的发现一致。
- **h=5 基本没有信号。**
- **dev 期 Sharpe 1–2，测试期只剩 0.2–0.5**：方向是在 dev 上选的，有明显衰减。

**可交易版本（测试期，h20）：**

| | Sharpe | 年化 | 最大回撤 |
|---|---|---|---|
| SPY 买入持有 | 1.40 | 21.9% | −18.8% |
| SPY TILT VOTE INTRA（100%±50%） | 1.38 | 23.6% | −20.4% |
| SPY MV C-LASSO INTRA | 1.43 | 29.0% | −20.2% |
| SPY MV 仅历史均值 | 1.20 | 26.2% | −27.2% |
| QQQ 买入持有 | 1.50 | 32.4% | −22.8% |
| QQQ TILT EW_DEV ALL | 1.52 | 35.1% | −27.9% |
| QQQ TILT VOTE INTRA | 1.50 | 35.1% | −24.2% |
| QQQ MV C-LASSO INTRA | 1.10 | 29.6% | −32.6% |
| QQQ B&H + LS VOTE（最高约 2x 杠杆） | 1.46 | 37.8% | −25.6% |

- TILT 的平均敞口约为 1，可以直接和买入持有比较，Sharpe 与买入持有持平（零假设 TILT 为 1.3–1.45）。
- MV C-LASSO 在 SPY 上好于买入持有，但在 QQQ 上明显变差，结论不稳健。
- 总体而言，**超额收益主要来自敞口和杠杆，择时本身不能稳定提高 Sharpe**。

## 5. 结论与建议
1. **因子有信息。** h20 时测试期 IC 比零假设高约 0.03，统计上稳健。但单因子的 Sharpe 量级只有 0.05–0.1，只能靠多因子合成，合成后约 0.4–0.5。
2. **推荐的处理流程：**
   - 时间聚合用 EWM 或 MA，窗口与预测期限匹配；
   - 标准化用 252 日滚动的稳健 z 分数（中位数/MAD），并做截尾；
   - 方向用 dev 冻结，或用扩张秩 IC；不要用扩张 OLS，也不要用 registry 里的先验方向；
   - 合成用等权或投票；
   - 仓位用 100%±tilt 的方式，或按 B&H + 小幅叠加。
3. **不推荐：**
   - 扩张窗口的统计量（会受水平漂移影响）；
   - 变化量类聚合；
   - 递归 OLS 得到的 μ̂ 仓位（存在机制性 Sharpe 偏差，R²_OS ≈ 0）；
   - PCA 和 PLS。
4. **噪声带中位数类因子等价于短期反转；偏度类因子（SKEW_DIFF、VP_MED_SKEW、LOGVOL_SKEW）无效。**
5. **多重检验：** 共 128 × 7 × 15 × 3 × 2 组单因子配置、9 × 3 × 2 × 2 × 3 组合成配置。所以只看与零假设对照后的均值差，不挑选单个最优配置。
