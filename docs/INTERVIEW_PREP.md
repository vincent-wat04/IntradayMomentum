# 面试准备 / Interview Prep：QQQ 日内噪声区间动量 + 风险叠加研究
# Intraday Noise-Area Momentum on QQQ + Risk Overlays — project walkthrough

> 用法：先背第 0 节（30 秒/2 分钟版本），再按第 1~3 节理解逻辑，第 6 节是高频 Q&A（中英），第 7 节是"诚实边界"——面试官追问时，**承认局限比硬撑更加分**。
> 所有数字都能在仓库 `outputs/` 里复现；背之前请自己重跑一次 `strategy_comparison.py → tearsheet.py`，面试时能说"我跑过"。
> 关于 AI 辅助：如果被问到工作流，如实说明用了 AI 编程助手写代码/跑实验，**你负责定假设、定判据、审结果**。别暗示全部手写。

---
## 0. 电梯陈述 / Elevator pitch

**30 秒（中文）**
我复现并改造了 Zarattini-Aziz-Barbon 2024 的日内"噪声区间"动量策略，用在 QQQ 上。重点不是调出一个漂亮的回测，而是建了一套**防自欺**的流程：先做机械审计（57 项检查，保证没有前视、成本和成交假设合理），所有改进都先写预注册文档再跑，用"同敞口对照 + 成对块自助 + Bonferroni"判定，失败的想法（GEX、止盈、恐慌因子、做空、期权备兑）全部如实否决。最终上线的是：日内 Sleeve + 一个按已实现波动调权的 QQQ 多头腿，在 moomoo 模拟盘跑；组合 Sharpe 1.36→1.56，MDD −26.6%→−19.8%，但我也明确这部分改善大多来自 2020 和 2022 两次波动尖峰，统计上不显著。

**30 sec (English)**
I replicated and extended the Zarattini–Aziz–Barbon (2024) intraday "noise-area" momentum strategy on QQQ. The point wasn't a pretty backtest — it was building a process that prevents self-deception: a mechanical audit (57 checks for look-ahead, costs, fills), pre-registered proposals before every experiment, and decisions based on matched-exposure controls, paired block bootstrap and Bonferroni. Most ideas failed and were rejected on the record (GEX, take-profit, fear factors, shorting, covered calls). What went live on a moomoo paper account is the intraday sleeve plus a QQQ long leg scaled by realized-vol percentile; the combo goes from Sharpe 1.36 → 1.56 and MDD −26.6% → −19.8%, but I'm explicit that most of that comes from two vol spikes (2020, 2022) and isn't statistically significant.

**2 分钟补充点 / 2-minute extras**
- 策略由两部分：① Sleeve（日内多空、隔夜空仓、β≈0）② QQQ 买入持有腿（提供 β）。组合 = 1×Sleeve + QQQ 腿。
- 研究主线三段：**审计 → 找 alpha 来源并证伪 → 风险管理（回撤）**。
- 最有价值的"产出"其实是**否决清单**和**方法论**，而不是最终参数。
- 工程：回测与实盘代码逐位一致验证（0.00e+00）、systemd 单次定时器、预检 22 项、仓位/购买力保护、发现并清除硬编码密码后重建仓库历史。

---
## 1. 研究弧线 / Research arc

### 阶段 A：复现 + 机械审计
- 论文策略：以开盘价为中心，用过去 14 天同一时刻的平均涨跌幅估计"噪声区间"（上下边界）；价格突破边界 → 顺势入场；到收盘平仓；用 VWAP 做追踪止损；用波动目标定仓位。
- 我先做了一个**审计工具集**：引擎 19 项（信号、成交、费用、日历、对齐）+ 数据 22 项（缺口、重复、时区、拆分）+ 其他共 57 项机械检查，并加了"未来数据投毒测试"（把未来数据改成极端值，信号不应变化）和"反向对照"。
- 发现：早期结果有"相位运气"（入场时间网格对齐到某个分钟时结果好看），说明单一参数点不可信 → 之后一律看参数面而非单点。

### 阶段 B：Sleeve 本身的改进（每个都预先写假设）
| 想法 | 做法 | 结果 | 教训 |
|---|---|---|---|
| 止损形态 | 去掉/保留空头止损；band + 硬尾 0.5 | **保留止损**，采用 band+尾 0.5 | 空头无上限风险，止损是风险控制不是 alpha |
| 排除周一 | 周一均值≈0（t=−0.23，p=0.23） | 保留排除，但**不计其测试集收益** | 假设来自看数据，必须降级 |
| 止盈 | 真实止盈槽位 + 安慰剂 | 安慰剂显示收益来自"少做空"，叠加 ΔSR=−0.003 | 先问"收益从哪来"，再问"赚多少" |
| GEX（做市商 gamma）跳过/加仓 | 预注册；高 GEX 日反而更好 | **否决**，60–85% 是敞口假象 | 条件分组前先控制敞口 |
| 波动调仓（Sleeve 内） | 指数缩放测试 | 无稳健改善 | — |

### 阶段 C：为什么 Sharpe 不是 2？长短、仅多头、反向 ETF
- 对比论文口径与自己的口径（成本、杠杆、多空），做了定义对账。
- 仅多头版本 / 反向 ETF（衰减）都不如多空版 → 否决。

### 阶段 D：组合与回撤管理（本项目后半部分）
1. **QQQ 腿的回撤才是组合回撤的主要来源**（Sleeve MDD −11.6%，QQQ −35.1%）→ 先试**静态再配比**（0.6 : 1.4）：Sharpe 1.404，MDD −15.9%，**与行情状态无关的回撤修复**。
2. **RV21 百分位降风险**（w = 0.5 + 1.0·(1−p)，p = 21 日已实现波动在过去 504 天中的因果百分位）：
   - 对比恐慌因子（VIX/VVIX/COR3M）：RV21 更好，因为恐慌因子携带风险溢价，降风险等于反向交易自己的 alpha。
   - 撤回过强结论：36 年 SPX 上 RV21 夏普改善为负（−0.093）；regime 叙事被证伪（剔除 COVID 后 SPX ΔSR −0.131；海外 2010 后 0/5 为正）。→ "**crash insurance，不是 regime alpha**"。
3. **权重敏感性 + 15 种组合**：lo=0（红圆）只在 2019–26 样本里最好，在 1992–2007 最差（SPX ΔSR −0.403）；lo=0.5（浅蓝）跨窗口稳定 → 选浅蓝，**不是因为它最高**，而是因为它最不容易在别的窗口翻车。
4. **Round 3 预注册三假设 + Bonferroni**（IV–RV 差、FOMC、COR1M）：0/3 通过；SPY 与 QQQ 的 Sleeve 相关 0.79，有效独立样本只有约 111/949 → SPY 不是独立的样本外验证。
5. **熊市诊断**：快崩 lo=0 更好，慢熊（2000–02）里红圆反而 −59.3%（B&H −49.1%）；机制 = 百分位对自身过去 504 天自适应 + hi>1 杠杆。
6. **期权备兑**（合成 Black-Scholes，对照"同 delta 静态组合"）：头条格 ΔSR=+0.032，四条预注册采纳标准全不满足 → 无值得采用的增厚；且没有真实期权链历史，只能说"合成定价下"。
7. **慢熊专项**（8 个预注册方案，4 个美股指数，1992–2026）：12-1 动量和 SMA200 通过全部规则，但去掉互联网泡沫后优势基本消失。
8. **系统化 v2**（7 个市场含 EWJ/EFA/EEM，31 年向前滚动选参，允许做空）：RV 与趋势**不独立**；200 日、0.5 **无统计证据**；做空一致有害；事前估计参数样本外不稳定 → 只留"12-1 动量 + e_off=0.5"作候选，**不上实盘**。

### 阶段 E：上线
- 选择"浅蓝圆"（lo=0.5, hi=1.5）：1×Sleeve + QQQ 腿，moomoo 模拟盘。
- 共享 CAPITAL_BASE = $500k（$1M 时最坏总敞口超过购买力）。
- QQQ 腿是 oneshot systemd timer（服务器内存紧，常驻进程曾引发故障）。

---
## 2. 方法论工具箱 / Methodology toolkit
| 工具 | 为什么 / Why |
|---|---|
| **机械审计 + 未来投毒测试** | 回测"没有前视"必须被证明，不是声明 |
| **预注册（先写判据后跑）** | 防 HARKing / 事后挑选；失败也留档 |
| **同敞口对照（matched-exposure control）** | 任何"降风险"类改进，先和"同平均敞口的静态组合"比，否则只是少持仓 |
| **成对循环块自助（L=10, 4000 次）** | 日收益有序列相关/波动聚集；成对比较削弱共同噪声 |
| **Bonferroni** | 同一轮多个假设时控制族错误率（例：3 假设 α=0.0167；8 方案 α=0.00625） |
| **有效独立样本数** | 两个相关 0.79 的资产不是两份证据 |
| **dev/test 切分 + 多窗口稳定性** | 不选最优格，选"邻域稳定"的格；报告最坏窗口 |
| **回测/实盘逐位一致性（replay equivalence）** | 标量实现 vs 向量实现差 0.00e+00 |
| **偏离记录（deviation log）** | 实验中发现的 bug 和口径变更全部写下来 |
| **置信区间 + 不显著就说不显著** | 面试官最信任这点 |

---
## 3. 实验目录 / Experiment catalog（一页速查）
| # | 实验 | 核心发现 | 一句话 |
|---|---|---|---|
| 1 | 机械审计 | 57 项通过；发现相位运气 | 单点不可信 |
| 2 | 止损/Monday | 保留止损；Monday 降级 | 假设来源决定可信度 |
| 3 | 止盈 + 安慰剂 | 收益来自"少空" | 先问来源 |
| 4 | GEX 跳过 | 否决，60–85% 敞口假象 | 控制敞口 |
| 5 | 恐慌因子降回撤 | 都输给 RV21 | 风险溢价型信号做不了风控 |
| 6 | IV–RV 错位/恐慌加仓 | 撤回 RV21 regime 叙事 | 勇于撤回 |
| 7 | Round 3（3 假设） | 0/3 | Bonferroni 有牙齿 |
| 8 | RV 权重敏感性/15 组合 | lo=0 最差窗口 −0.403 | 选稳健不选最高 |
| 9 | 熊市诊断 | 快崩有用，慢熊无用 | 机制 > 数字 |
| 10 | 备兑期权 | ΔSR +0.032，四条全否 | 合成定价限制 |
| 11 | 慢熊专项 | 趋势类保险，非 alpha | 一个事件依赖 |
| 12 | 系统化 v2 | 参数无证据、做空有害 | 估计参数反而更差 |

---
## 4. 工程与运维 / Engineering
- 实盘：moomoo OpenD 模拟盘；Sleeve 常驻服务，QQQ 腿 oneshot 定时器（ET 15:55–15:59 窗口、北京时间两个触发点以兼容夏令时）。
- 保护：交易代码守卫（避免 Sleeve 读到 QQQ 腿持仓而平仓）、购买力 85% 上限、最小调仓 0.3%、单次最多 3 笔订单。
- 一致性：leg 标量 vs 向量化 1008 天差 0；Sleeve replay 13 个配置差 0。
- 事故（面试可以主动讲，体现成熟度）：
  1. 服务器内存不足导致常驻进程故障 → 改 oneshot。
  2. 我（助手）一次研究拉取消耗了历史 K 线配额 228/297 → 加 bootstrap 保护并验证新接口不消耗配额。
  3. 发现仓库历史里有硬编码 root 密码 → 重建全新历史、加密钥扫描（构建时硬失败）、建议轮换并改密钥登录。
  4. 审计中发现的口径 bug（预检真值表写死默认值、期权滚动日丢股票收益等）→ 全部写入偏离记录。

---
## 5. 数字速查 / Numbers cheat-sheet
| 项目 | Sharpe | CAGR | MDD |
|---|---|---|---|
| Sleeve | 0.954 | — | −11.6% |
| QQQ B&H（2019–26） | 0.994（dev 0.662 / test 1.496） | — | −35.1% |
| 组合 1:1 | 1.360 | 38.5% | −26.6% |
| 静态再配比 0.6:1.4 | 1.404 | — | −15.9% |
| **浅蓝（上线）** | **1.563**（dev 1.10 / test 2.05） | 39.3% | −19.75% |
| 红圆 lo=0/hi=2 | 1.611 | — | −16.6% |
- 浅蓝：换手约 6.9 次/年，β=0.766，α=17.6%（t=3.64，**机械来自更低 β，不是 RV 创造了 alpha**）。
- 约定：项目内 Sharpe/α 早期沿用"未减无风险利率"口径；后期研究（慢熊/系统化）用超额收益。**面试时主动说明**。

---
## 6. Q&A（中英）

### A. 策略与动机 / Strategy & motivation

**Q1. 为什么选这个策略？ / Why this strategy?**
- 中：它有公开论文、规则透明、可完全机械化，适合作为"怎么检验一个声称有效的策略"的载体。我关注的是检验流程，而不是挖新 alpha。
- EN: It's published, transparent and fully mechanical — a good vehicle to demonstrate how to *test* a claimed edge. My focus was the validation process, not hunting for a new alpha.

**Q2. 策略的经济逻辑？ / Economic rationale?**
- 中：日内动量来自①流动性提供者/对冲者的被动再对冲（比如 gamma 对冲在趋势日放大波动）②信息逐步扩散与羊群行为③指数 ETF 的尾盘再平衡与杠杆 ETF 再平衡。边界"噪声区间"过滤掉没有方向性的噪声。我**测试过 GEX 作为机制解释，但没有得到支持**——所以我不声称知道真正的机制。
- EN: Candidates: dealer hedging flows, gradual information diffusion/herding, and end-of-day rebalancing (incl. leveraged ETFs). The noise area filters undirected noise. I *tested* the GEX channel and found no support, so I don't claim to know the true mechanism.

**Q3. Sharpe 为什么只有 ~1 而不是论文里的 2 以上？ / Why not the paper's Sharpe?**
- 中：对账后主要差异：口径（是否扣无风险利率、杠杆）、成本、样本区间与标的（SPY vs QQQ）、多空限制。我先把定义对齐再比较，而不是调参去追数字。
- EN: After reconciling definitions: rf treatment/leverage, costs, sample window and underlying, long/short constraints. I aligned definitions first rather than tuning to chase the number.

### B. 回测陷阱 / Backtest pitfalls

**Q4. 你怎么证明没有前视偏差？ / How do you rule out look-ahead?**
- 中：三层：①机械审计（对齐、时区、信号只用 t 及以前）；②**未来数据投毒**——把未来数据替换成极端值，信号必须不变；③**实盘逐位对账**——向量化回测与实盘标量代码输出差 0.00e+00。
- EN: Three layers: mechanical audit; future-poison test (replace future data with extreme values — signals must not move); and bit-for-bit equivalence between the vectorized backtest and the live scalar code (diff = 0).

**Q5. 你怎么处理多重检验？ / Multiple testing?**
- 中：每轮实验先写预注册文档，固定假设数和判据；用 Bonferroni；并把**所有**格子都报告，不挑最优。Round 3 里三个假设 0/3 通过，就如实写 0/3。另外相关资产不是独立证据（SPY–QQQ Sleeve 相关 0.79，有效独立样本约 111/949）。
- EN: Pre-register each round, fix the number of hypotheses and decision rules, apply Bonferroni, report every cell not the best. Round 3: 0/3 passed, reported as such. Correlated assets aren't independent evidence (SPY–QQQ sleeve corr 0.79 → effective n ≈ 111 of 949).

**Q6. 最大的过拟合风险在哪？ / Biggest overfitting risk?**
- 中：①浅蓝参数选择发生在 2019–26 的窗口内；②只有两次大波动尖峰；③我看过很多结果再做选择。缓解：选"跨窗口稳定"的格而非最优格，报告最坏窗口（红圆在 1992–2007 最差 −0.403，所以淘汰）；但我**不声称统计显著**。
- EN: Parameter choice happened within the 2019–26 window; there are only two big vol spikes; and I looked at many results before choosing. Mitigation: pick the cell that's stable across windows rather than the highest, report worst windows (the red circle fails in 1992–2007), and I do **not** claim statistical significance.

**Q7. 什么是同敞口对照？为什么需要？ / What is a matched-exposure control and why?**
- 中：降风险类改进（把仓位调低）自然降低回撤，这不是择时能力。所以我拿"同平均权重的静态组合"做对照；ΔSR 才是择时的净贡献。
- EN: De-risking trivially cuts drawdown by holding less. The control is a static portfolio with the same average weight; the Sharpe difference isolates timing skill.

**Q8. 为什么用块自助而不是普通 t 检验？ / Why block bootstrap?**
- 中：日收益有波动聚集与序列相关，iid 假设会低估方差。成对（同一重采样下比较两条序列）能消去共同噪声。块长 10。
- EN: Volatility clustering and serial correlation violate iid; paired resampling cancels common shocks. Block length 10.

### C. 风险与组合 / Risk & portfolio

**Q9. 为什么 RV 降风险有用？为什么你后来说它不是 alpha？ / Why does RV de-risking help, and why isn't it alpha?**
- 中：波动具有持续性，尖峰后的几天风险高；降仓位降低回撤。但夏普改善对同敞口对照大多不显著（ΔSR +0.03~+0.25、CI 多含 0），约 25–40% 的改善来自 COVID；36 年 SPX 上改善为负。所以它是"尾部保险"。
- EN: Vol is persistent, so cutting exposure after spikes lowers drawdown. But Sharpe gains versus matched controls are mostly insignificant, 25–40% come from COVID, and over 36 years of SPX the gain is negative. It's tail insurance, not alpha.

**Q10. 为什么选 lo=0.5 而不是 lo=0（红圆）？ / Why lo=0.5, not 0?**
- 中：红圆在 2019–26 最高（Sharpe 1.611），但在 1992–2007 窗口最差（SPX ΔSR −0.403）。浅蓝跨窗口更稳（6 个窗口平均 ΔSR +0.055，最差 −0.172）。我选**稳健**。
- EN: The red circle wins only in 2019–26 and is worst in 1992–2007. The light-blue is stable across windows (mean +0.055, worst −0.172). I optimized for robustness.

**Q11. 熊市里 lo=0 不是更应该躲吗？ / Shouldn't lo=0 handle bears better?**
- 中：快崩里是（COVID：B&H −33.9%，红圆 −6.8%，浅蓝 −20.9%）；慢熊不是（2000–02：B&H −49.1%，红圆 −59.3%）。因为百分位相对自己过去 504 天，持续的温和高波动会让百分位"适应"，权重回到 >1，叠加 hi=2 杠杆。**不是 lo 的问题，是百分位自适应 + 杠杆。**
- EN: In fast crashes yes (COVID: B&H −33.9%, red −6.8%, light-blue −20.9%); in slow bears no (2000–02: B&H −49.1%, red −59.3%). The percentile is relative to the last 504 days, so persistent moderate vol "adapts away" and weights drift above 1, compounded by hi=2 leverage.

**Q12. 慢熊你怎么处理？结论？ / How would you handle slow bears?**
- 中：需要价格趋势信号而不是波动信号。12-1 动量和 SMA200 在预注册规则下通过，但去掉 2000–02 后优势基本消失，夏普基本持平，主要是降回撤，代价是错过反弹。参数（200、0.5）没有统计证据，向前滚动选参在 7 个市场里 5 个输给固定规则。做空一致有害。**我没有上实盘。**
- EN: Need a price-trend signal, not vol. 12-1 momentum and SMA200 pass the pre-registered rules, but drop 2000–02 and the edge mostly disappears: Sharpe flat, drawdown lower, rebounds missed. The parameters (200, 0.5) have no statistical evidence; walk-forward selection lost to fixed rules in 5 of 7 markets; shorting hurts consistently. I did not deploy it.

**Q13. RV 与趋势独立吗？ / Are RV and trend independent?**
- 中：不独立。高波动日 38%~63% 处于 SMA200 之下，低波动日仅 6%~22%（杠杆效应），Spearman −0.26~−0.52。叠加后夏普贡献小于相加（DiD −0.02~−0.04），RV 叠在趋势上主要增加换手（约 3.4 倍）。
- EN: No. 38–63% of high-vol days are below SMA200 vs 6–22% of low-vol days (leverage effect), Spearman −0.26 to −0.52. Combined Sharpe contribution is sub-additive (DiD −0.02 to −0.04); layering RV on trend mostly adds turnover (~3.4×).

**Q14. 你为什么说 QQQ 腿的 α 不是真 α？ / Why is the QQQ-leg alpha not real alpha?**
- 中：浅蓝的 β=0.766，α=17.6%（t=3.64）是回归相对 QQQ 得出的；降低 β 后，Sleeve 自身的 α 占比被抬高，是**机械结果**，不是 RV 创造了 alpha。
- EN: With β=0.766 the regression alpha vs QQQ rises mechanically because the Sleeve's alpha is weighted more heavily; RV did not create alpha.

### D. 期权 / Options

**Q15. 能用期权增厚吗？ / Can options enhance returns?**
- 中：测了备兑看涨（合成 BS，σ=m×VXN，3% 价差+手续费，2009–2026，对照同 delta 组合）。头条格 ΔSR +0.032，CI [−0.145, +0.200]，四条预注册采纳标准全部不满足；下行 β 0.82 vs 上行 0.72，没有保护；牛市成本大。与 Israelov-Nielsen 一致。局限：无真实期权链历史、合成定价高估权利金、忽略偏斜。
- EN: Tested covered calls (synthetic BS, σ = m × VXN, 3% spread + commissions, 2009–2026, matched-delta control). Headline ΔSR +0.032, CI [−0.145, +0.200], all four pre-registered adoption criteria fail; downside β 0.82 vs upside 0.72 (no protection); costly in bulls. Consistent with Israelov & Nielsen. Caveats: no real chain history, synthetic pricing overstates premium, skew ignored.

### E. 工程 / Engineering

**Q16. 怎么保证实盘与回测一致？ / Live-backtest parity?**
- 中：两套实现（向量化回测、标量实盘），对 1008 天逐位比较差 0.00e+00；Sleeve 13 个配置 replay 差 0；预检 22 项（配置、购买力、交易代码守卫等）。
- EN: Two implementations (vectorized vs scalar); 1008 days compared bit-for-bit (diff 0); Sleeve replayed over 13 configs; 22 preflight checks.

**Q17. 讲一个你踩的坑 / A mistake you made?**
- 中：①曾把 52.9bp 的对账差错归因于 IV 标价，后来发现是分母口径，已在偏离记录里更正；②研究拉取消耗了历史 K 线配额 228/297，之后加了保护；③硬编码密码被发现，重建历史并加扫描。我的做法是**当天披露、修复、写进文档**。
- EN: (1) I mis-attributed a 52.9bp reconciliation gap to IV marking — it was the denominator convention, corrected in the deviation log. (2) A research pull burned 228/297 of the history-kline quota; added guards. (3) A hard-coded password was found; rebuilt history and added a build-time secret scan. I disclose, fix and document immediately.

**Q18. 你的延迟/成交假设？ / Execution assumptions?**
- 中：Sleeve 用分钟线 + 保守成本；QQQ 腿在收盘前窗口（ET 15:55–15:59）市价单。模拟盘不代表真实滑点——我用实测成交均价对比信号价，并保留了成本敏感性。
- EN: Minute bars with conservative costs for the sleeve; market orders in the closing window for the leg. Paper trading doesn't capture real slippage — I compare fills to signal prices and keep a cost sensitivity.

### F. 压力问题 / Hard questions

**Q19. 如果让你把这个策略交给 PM，你会怎么说？ / Pitch it to a PM?**
- 中：不会说"有 alpha"。会说：①Sleeve 与市场几乎无关（β≈0），是低相关的分散化来源；②QQQ 腿上的 RV 降风险是低成本的尾部保险，不是收益来源；③全部结论有明确局限和否决清单；④建议先小资金、纸面验证 6–12 个月，再看真实滑点。
- EN: I wouldn't claim alpha. I'd say: the sleeve is a low-correlation diversifier (β≈0); the RV overlay is cheap tail insurance, not a return source; every conclusion comes with limits and a rejection list; start small and paper-trade 6–12 months to measure real slippage.

**Q20. 你的结果统计上显著吗？ / Is it significant?**
- 中：整体看**不显著**：RV 叠加的 ΔSR 区间多含 0；慢熊只有约 2 个独立事件；备兑 t=1.71。我宁可报告"证据不足"也不包装。
- EN: Mostly no — RV-overlay ΔSR intervals mostly include zero; slow bears give ~2 independent events; covered call t = 1.71. I'd rather report "insufficient evidence" than dress it up.

**Q21. 如果重做你会改什么？ / What would you do differently?**
- 中：①一开始就把"开发/测试/最终留出"三段切好，并只在留出段碰一次；②更早拿到真实期权链和更长历史（1990 年之前）；③从一开始就用超额收益口径；④把成本敏感性写进每个预注册。
- EN: Split dev/test/final holdout from day one and touch the holdout once; get real option chains and pre-1990 history earlier; use excess-return Sharpe from the start; build cost sensitivity into every pre-registration.

**Q22. 为什么相信 SMA200 / 12-1 动量？ / Why trust them?**
- 中：不"信"。长度不敏感、夏普≈持平，价值是回撤；我检验了向前滚动选参，结果不稳；因此只保留为"预先固定的文献标准值"候选，不上线。
- EN: I don't "trust" them: length-insensitive, Sharpe ≈ flat, value is drawdown. Walk-forward selection was unstable, so I keep them only as a pre-fixed literature-standard candidate, not deployed.

**Q23. 容量/规模？ / Capacity?**
- 中：QQQ 流动性极好，$500k 基数远低于容量；但日内策略对冲击和排队敏感，规模放大要重新测滑点。Sleeve 是规模不变的比例策略，但真实成交不是。
- EN: QQQ is extremely liquid; $500k is far below capacity. Intraday edges are sensitive to impact, so scaling requires re-measuring slippage.

**Q24. 为什么不直接做空 / 反向 ETF？ / Why not short or use inverse ETFs?**
- 中：做空在趋势类上一致有害（7 个市场里 SMA200 e<0 全为负，原因是反弹打穿空头 + 融券/股息成本）；反向 ETF 日再平衡衰减。Sleeve 本身已含日内空头，单独的隔夜空头没有证据支持。
- EN: Shorting is consistently harmful for trend rules (SMA200 e<0 negative in all 7 markets: rebounds stop out shorts plus borrow/dividend costs); inverse ETFs decay from daily rebalancing; the sleeve already has intraday shorts.

**Q25. 一句话总结你学到什么 / One-line takeaway?**
- 中：**好的研究不是找到最好的参数，而是能说清楚哪些东西不能信、为什么。**
- EN: Good research isn't finding the best parameter; it's being able to say what you can't trust and why.

---
## 7. 诚实边界 / Honest boundaries（被追问时的底线）
1. 样本短：Sleeve 与组合只有 2019–2026；两次大波动尖峰。
2. 浅蓝的选择发生在同一窗口内，虽然做了跨窗口稳定性检查，仍不是干净的样本外。
3. 模拟盘 ≠ 真实成交；没有测过真实滑点/冲击。
4. 期权结论建立在合成定价上。
5. 早期 Sharpe/α 沿用未减无风险利率的口径；后期研究用超额收益。
6. 趋势保护依赖少数几次熊市；去掉 2000–02 后几乎无优势。
7. 没有 1990 年以前日线数据。

## 8. 可以主动提的"加分点" / Proactive highlights
- 我保留了**否决清单**，并撤回过自己的结论（RV21 regime 叙事）。
- 预注册 + Bonferroni + 同敞口对照 + 偏离记录。
- 回测实盘逐位一致、安全守卫、事故复盘。
- 研究库与实盘库分离，构建时自动扫描凭证。

## 9. 反问面试官的好问题 / Questions to ask back
- 你们怎么管理研究中的多重检验？有没有统一的预注册或"假设注册表"？
- 回测与实盘一致性是怎么检验的？
- 对"没有显著性但风险管理有价值"的信号，你们如何决策上线？
