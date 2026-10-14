# noise-momentum-research

QQQ 日内噪声区间动量策略的**完整研究记录**：79 个实验脚本、57 项机械化审计、
14 篇研究文档、49 张图。实盘代码在 `noise-momentum-live`。

这份记录刻意**保留了失败的实验和被推翻的结论**。做量化研究，
"什么不work、以及为什么"比最终那个配置更值钱。

---

## 0. 快速开始

```bash
pip install -r requirements.txt
export ETFT_DATA=/path/to/data            # 原始数据不入库（约 1.2GB），见 data/README.md
# 实盘库放在本库同级目录（noise-momentum-live/），或 export LIVE_SRC_ROOT=/path/to/noise-momentum-live

# ① 最重要的两个检验：实盘代码 == 已审计的回测（必须全部 0.00e+00）
python tools/replay_equivalence.py        # Sleeve：1948 天 × 390 bar，13 个配置
python tools/leg_equivalence.py           # QQQ 腿：全历史抽样 + 最近 120 天 + 未来数据投毒

# ② 三层机械化审计（在 studies/ 下运行；产物落在 studies/outputs/，已 .gitignore）
cd studies
python audit/audit_engine.py              # 19/19  引擎层
python audit/audit_data.py                # 22/22  数据层
python audit/audit_factors.py             # 16/16  因子层（需要 data/xs 横截面数据）

# ③ 复现最终交付图（有先后顺序：后者读前者产生的曲线文件）
python strategy_comparison.py             # 约 10 分钟：12 个版本 × 30 相位 -> outputs/full/intraday/*.pkl
python tearsheet.py                       # 业绩总览（含 QQQ 腿 RV 降风险）
python combo_comparison.py                # 静态配比 × RV 降风险 的组合对比
python sens_rv_weights.py                 # RV 权重敏感性（参数全部可用命令行改）
```

**目录**：`studies/` 实验脚本（扁平；含 vendor 的 `etft/` 引擎、`audit/`、`paths.py`）、
`docs/` 研究文档、`outputs/` 已固化的图与表、`tools/` 等价性检验与需要连服务器的工具。

**路径**：所有数据路径从 `studies/paths.py` 取，由 `ETFT_DATA` 控制；没有任何写死的本机目录。
**凭证**：仓库里没有任何凭证。`tools/crosscheck.py`、`monday_control.py` 需要连服务器，
凭证通过环境变量或不入库的配置文件提供（见实盘库 `ops/sshx.py` 头部）。

## 1. 最终结论

**部署的组合** = 1 × **QQQ 腿**（按 RV21 百分位调权 w ∈ [0.5, 1.5]）+ 1 × **Sleeve**（多空 + band 止损 + 硬尾 k=0.5 + 排除周一 + voltarget 4x）。
两条腿共用同一个资金基数；详见实盘库 README。

| 口径（全样本 2019-2026，30 相位均值、TQQQ 成本） | Sleeve | QQQ 买入持有 | 组合 1:1 | **组合 + QQQ 腿 RV 降风险** |
|---|---|---|---|---|
| 年化收益 | 12.19% | 23.22% | 38.50% | 39.28% |
| 年化波动 | 12.91% | 23.88% | 26.52% | 22.86% |
| **Sharpe（绝对）** | 0.954 | 0.994 | 1.360 | **1.563** |
| 最大回撤 | −11.61% | −35.12% | −26.56% | **−19.75%** |
| Calmar | 1.049 | 0.661 | 1.450 | 1.989 |
| 分期 Sharpe dev / test | 0.70 / 1.20 | 0.66 / 1.50 | 0.93 / 1.92 | 1.10 / 2.05 |
| β（对 QQQ） | −0.030 | — | 0.970 | 0.766 |
| 与 QQQ 相关性 | −0.055 | — | 0.874 | 0.800 |

**结论的正确读法**

1. Sleeve 不是"比指数更好的策略"，而是 **β≈0、α≈13%、与指数相关性≈0 的分散化来源**。
   Sharpe 对规模齐次，无法靠加杠杆改善 QQQ；唯一的办法是加入低相关正收益资产。
2. **QQQ 腿 RV 降风险是"崩盘保险"，不是稳健的 alpha。** 相对同平均敞口的纯静态对照 ΔSharpe +0.20，
   但自助区间 [−0.01, +0.42]；剔除 2020 年崩盘后约缩水 25%~40%；
   在 2019 年之前的窗口里两个指数都是负的（NDX 1992-2007 −0.06，SPX −0.17）。
   **参数是看过 2019 年之前的 SPX/NDX 结果后选定的，与评价窗口有重叠，不是独立样本外。**
   它需要平静期最高 1.5 倍的 QQQ 杠杆。
3. 不需要杠杆的更保守选择是静态再配比 0.6:1.4（Sharpe 1.40、MDD −15.9%），它不依赖任何体制假设。
4. **最终仲裁是模拟盘的样本外**（Sleeve 自 2026-10-05，QQQ 腿自 2026-10-08）。

## 2. 实验索引

### A. 框架与数据

| 脚本 | 做了什么 / 发现 |
|---|---|
| `index_timing_framework.py` `analyze_full.py` | 157 因子 × 多模型的择时框架，全量数据跑通 |
| `run_full_models.py` `run_full_extended.py` | base / extended / macro / linear / RAVEN 五套模型 |
| `consolidate_single_symbol.py` | 发现**所有策略都是单标的**，促成后续横截面方向 |

### B. 日频因子与择时模型 → `docs/FULL_RESEARCH_REPORT.md` `docs/TIMING_TRANSFORMS.md`

| 脚本 | 做了什么 / 发现 |
|---|---|
| `factor_sharpe.py` `factor_sharpe_report.py` | 单因子多空 Sharpe 中位数仅 ~0.1 |
| `timing_transform_study.py` | **发现递归 OLS 的下注规模偏差**；8 个 horizon 的 IC 呈 U 形 |
| `timing_combination_study.py` `timing_model_strategy.py` | 模型→策略的映射；COMBO 最优 |
| `placebo_transform.py` | **发现时间平移式的 null 不成立**，换用置换检验 |
| `macro_factor_study.py` `multifrequency_models.py` `raven_*.py` | 宏观/多频/RAVEN 复现，均无显著 alpha |

### C. 横截面 → `docs/PART3`~`PART6`

| 脚本 | 做了什么 / 发现 |
|---|---|
| `xs_factor_library.py` `xs_families.py` | 因子库从 32 扩到 77（补 microstructure）|
| `xs_pipeline_v2.py` `xs_models.py` | ElasticNet 与 COMBO 并列最优 |
| `xs_attribution.py` `xs_turnover.py` | 单因子强、复合弱的归因阶梯；换手不是主因 |
| `fundamental_factors.py` | 基于 SEC XBRL 的 9 个 PIT 基本面因子，信号混合 1.436 |
| `pdf_xs_replication.py` `pdf_factor_test.py` | 研报里 10 个因子**只有 2 个能复现** |
| `protocol_universe_compare.py` | 1350 只universe；sp600 小盘最好 1.807 |
| `cn_*.py` | A 股横截面；受 moomoo 额度限制只跑到 125/1499 |

### D. 组合与杠杆 → `docs/PART2`

| 脚本 | 做了什么 / 发现 |
|---|---|
| `portfolio_construction.py` `leveraged_portfolio.py` `leverage_frontier.py` | **beta 对冲把 sleeve 打死了**；Calmar 对杠杆近乎平坦 |
| `qqq_overlay_check.py` | 叠加层相对"敞口匹配基准"只多 +0.034 |

### E. 日内策略主线 → `docs/PART8`~`PART11`

| 脚本 | 做了什么 / 发现 |
|---|---|
| `intraday_battery.py` | 多个日内策略横评；**查出一根 bar 的 look-ahead**（Sharpe 高得不合理）|
| `sota_tqqq.py` | 修正 PART8 的**三个 TQQQ 错误**（成本记在 3 倍名义额、用后复权价算单股成本、测的不是 SOTA）|
| `grid_phase_ensemble.py` | **发现"相位运气"**：论文相位是 30 个里的第 90 百分位 |
| `convexity_strategy.py` | 凸性收益真实但不可条件化（置换 p=0.092）|
| `long_short_decomposition.py` | 多空 vs 只做多的分解；只做多独立更好但叠加更差 |
| `inverse_etf_study.py` | SQQQ/QID/PSQ 跟踪极好但成本 2.7~6 倍，把叠加贡献吃到零 |
| `short_leg_diagnosis.py` | **空头低胜率的根因是止损不是信号**：64.6% 被止损、胜率 10.1%；持到收盘胜率 77.6% |
| `improvement_tests*.py` | 论文 §4.2 NR4（置换 p=0.131）、§4.4 VM=1.5、长期信号、fast alpha、量能确认 —— **全部不成立** |
| `reconcile_numbers.py` | 对齐 "Sharpe 2.0" 与 "0.3~0.9" 两套口径的差异 |

### F. 止损设计 → `docs/PART11`

| 脚本 | 做了什么 / 发现 |
|---|---|
| `stop_removal_study.py` | 去掉空头止损：逐笔均值几乎不变，但最差单笔 −2.00%→−4.63% |
| `stop_design_study.py` | 8 个止损族 × 30 相位全扫 |
| `stop_significance.py` | 配对区块自助；**"不显著"不等于"不该换"** |
| `stop_hard_tail.py` | 硬尾止损：ΔSharpe ±0.003、约 2.5 次/年 —— 近乎零代价的结构性保护 |
| `stop_decision.py` | 最终选 band+尾0.5（非统计理由：换手、亏损上限、regime 稳健性）|

### G. 止盈 / 同向再开仓 → `docs/PART12`

| 脚本 | 做了什么 / 发现 |
|---|---|
| `takeprofit_diag.py` | **先测条件期望再谈策略**：空头到 +0.5~1σ 后平均回吐 13~30bp，dev/test 都显著（按天聚类）|
| `takeprofit_study.py` | 真止盈（独立槽，不被 `reentry="hold"` 抑制）；独立 Sharpe 全面提升 |
| `takeprofit_placebo.py` | **安慰剂救了我一次**：随机平空 + 空头规模缩放两个对照说明收益来自"少做空"而非择时；叠加贡献 −0.003 |
| `takeprofit_gex.py` | GEX 条件化止盈：gamma 机制预测"高 GEX 日最有效"，**实测恰好相反** |

### H. GEX 与波动率 → `docs/PART12` §6

GEX 一共试了 **8 种用法，全部否决**。唯一复现成功的是"GEX 预测日内波动幅度"
（四分位 |open→close| 比值 0.42~0.56、样本外 R² 0.009→0.035、DM t 最高 +3.88），
但它换不成策略收益（dev +0.166 / test −0.100）。

| 脚本 | 做了什么 / 发现 |
|---|---|
| `gex_false_breakout.py` | 过滤假突破：z=+0.08，无效 |
| `gex_scaling_raw.py` | 原始值（非分位）做仓位缩放：p=0.15~0.29 |
| `gex_lag_study.py` | **滞后形状非单调** → 原先的 t−2"信号"是 5 个里挑最好的 |
| `gex_reversion_study.py` | GEX 条件化的反转，全负 |
| `gex_vol_forecast.py` | 波动率预测三层检验（唯一成立的那个）|
| `gamma_proxy_study.py` | 论文 §4.5 的 RSI(5) 代理：falsification 通过（corr 0.49~0.61）但 β 不复现（+0.088 vs 论文 −3.25）|
| `vol_sizing_exponent.py` | 反向 vol targeting 扫描：β>0 单调更差；**凸性是同日属性、vol targeting 是 regime 层** |

### I. 版本对照与交付 → `docs/PART13`

| 脚本 | 做了什么 / 发现 |
|---|---|
| `strategy_comparison.py` | 12 个版本 × 4 因子的完整对照（Sharpe/年化/Calmar/分边胜率盈亏比）|
| `alpha_beta_decomp.py` | α/β/alpha Sharpe/绝对 Sharpe；**corr(β, 叠加贡献) = −0.905** |
| `tearsheet.py` | 客户演示用业绩总览 |

### J. 第三轮预注册与 RV 降风险 → `docs/PROPOSAL_*.md` `docs/SENSITIVITY_rv21_weights.md`

| 脚本 | 做了什么 / 发现 |
|---|---|
| `gex_skip_study.py` | 规避极高 GEX 日（预注册）：**否决**。高 GEX 日反而更好，但 60~85% 是敞口假象；并修了自己的判定 bug |
| `fear_derisk_study.py` | 恐慌族因子（VIX/COR3M/VVIX）降回撤：**全部输给 RV21**。它们携带风险溢价，降风险等于反向交易自己的 alpha |
| `iv_rv_panic_study.py` | 隐含-已实现波动错位、恐慌后加仓；撤回上一轮对 RV21 的"regime"叙事 |
| `round3_study.py` + `round3_common.py` | 三个预注册假设 + Bonferroni：**0/3 通过**；RV21 体制检验（剔 COVID 后标普 ΔSR −0.131，海外 0/5 为正） |
| `sens_rv_weights.py` | RV 降风险权重敏感性：lo=0 是 Sharpe 最差的一行；抬到 0.25~0.5 后 6/6 样本改善（仍在噪声内） |
| `combo_comparison.py` + `combo_lib.py` | 静态再配比 × RV 降风险 15 个组合；每个 RV 组合配"同平均敞口纯静态对照" |
| `bear_regime_diag.py` | 熊市诊断（因果回撤≤−20%）：快崩 lo=0 更好；慢熊（互联网泡沫）红圆反而更差——不是 lo 的问题，而是 hi>1 的杠杆 + 分位数自适应 |
| `slowbear_study.py` + `slowbear_verdict.py` + `slowbear_robust.py` + `plot_slowbear.py` | 慢熊行情处理（预注册 8 方案，1992-2026，SPX/NDX/RUT/NYA）：**趋势200 与 12-1 动量通过全部规则**（泡沫段峰→谷 −54%→−30~−33%），但夏普持平、去掉 2000-02 后优势基本消失；RV/回撤刹车/波动目标/曲线倒挂均否决 |
| `systematic_trend_rv.py` + `plot_systematic.py` | RV×趋势系统化组合（预注册 v2，7 市场含 EWJ/EFA/EEM，31 年向前滚动选参，允许做空）：**RV 与趋势不独立**（高波动日 38~63% 处于下跌趋势）；200 日/0.5 无统计证据；做空一致有害；候选仅 12-1 动量 e=0.5 |
| `overwrite_study.py` | 备兑看涨增厚（合成 BS 定价，预注册，对照"同 delta"）：ΔSR +0.032，四条采纳标准全不满足；无真实期权链，结论限于合成定价 |

### K. 实盘化 → 实盘库 `noise-momentum-live`

| 文件 | 做了什么 |
|---|---|
| `tools/replay_equivalence.py` | Sleeve 的 `decide()` 逐 bar 回放 vs 参考实现，13 个配置 0.00e+00 |
| `tools/leg_equivalence.py` | QQQ 腿标量实现 vs 向量化实现：抽样 1008 个交易日 0.00e+00，并做未来数据投毒 |
| `tools/crosscheck.py` / `monday_control.py` | 拉服务器成交与回测对账 / 离线重放"不排除周一"的对照组（需连服务器） |

## 3. 文档地图

| 文档 | 内容 |
|---|---|
| `AUDIT.md` | 57 项机械化审计的方法与结果 |
| `LIVE_SYSTEM.md` | 实盘系统设计（**历史文档**，已被实盘库 README 取代） |
| `FULL_RESEARCH_REPORT.md` `TIMING_TRANSFORMS.md` `FACTOR_CATALOG.md` | 日频阶段 |
| `PART2`~`PART7` | 仓位/横截面/归因/基本面/小盘/策略排名 |
| `PART8` | 日内 TQQQ 与策略横评（**开头有勘误指向 PART9**）|
| `PART9` | TQQQ 三个错误的修正、凸性、口径对齐 |
| `PART10` | 审计发现与相位运气 |
| `PART11` | 空头腿诊断与各类改进尝试 |
| `PART12` | 止盈、同向再开仓、GEX wall |
| `PART13` | 12 版本对照、α/β 分解、**§7 是对 §3.2 的更正** |
| `PROPOSAL_gex_skip.md` | 预注册：规避极高 GEX 日（否决）。**先读它，自己想一遍该设哪些对照，再看脚本** |
| `PROPOSAL_fear_derisk.md` | 预注册：恐慌族因子降回撤 |
| `PROPOSAL_iv_rv_panic.md` | 隐含/已实现波动错位与恐慌后加仓 |
| `PROPOSAL_round3.md` | 三个预注册假设 + Bonferroni + RV21 体制检验 |
| `INTERVIEW_PREP.md` | 研究思路梳理、实验目录与中英文 Q&A（面试准备） |
| `PROPOSAL_systematic.md` | 预注册 v2 + 结果：RV×趋势 机械组合、参数证据、做空 |
| `PROPOSAL_slowbear.md` | 预注册 + 结果：慢熊行情的处理 |
| `PROPOSAL_overwrite.md` | 预注册：备兑看涨期权增厚（含偏离记录） |
| `SENSITIVITY_rv21_weights.md` | RV 降风险权重参数敏感性 |
| `COMBO_comparison.md` | 静态再配比 × RV 降风险 组合对比 |

## 4. 方法学：这个项目真正学到的东西

1. **读代码找不出记账 bug。** 我靠读代码漏掉了至少 5 个"看起来很正常"的错误。
   能用的是四种机械化手段：未来数据投毒、日内 bar 投毒、独立重写对账、oracle 对照。
   每个检验都要配一个**反向对照**（故意失败的用例），证明它有分辨力。
   见 `studies/audit/harness.py`。
2. **独立 Sharpe 会骗人。** 任何削弱空头腿的改动（止盈、缩规模、只做多）
   都会抬高独立 Sharpe、同时压低叠加贡献 —— 因为空头腿就是这个 sleeve 的对冲性。
   `corr(β, 叠加贡献) = −0.905`。决策必须看部署口径。
3. **匹配敞口的安慰剂对照不可省。** 止盈看上去是本项目最大的改进
   （dev 0.661→1.134、test 0.963→1.181），直到对照显示"随机平空"和
   "空头规模×0.6"能拿到同样甚至更好的结果。
4. **先测条件期望，再谈策略参数。** 止盈能不能赚钱只取决于
   "到达 +x·σ 之后剩下那段是正还是负"，这个量不依赖任何策略实现。
   先把它测清楚，省掉一整片参数扫描。
5. **相位运气是真的。** 检查时点整体平移 1 分钟就是一个新"相位"，
   30 个相位的 Sharpe 标准差 0.216，论文那个相位在第 90 百分位。
   单相位结果不可信。
6. **因子边际效应必须在"其他因子取推荐值"的条件下估计。**
   我曾把含已否决因子（止盈）的组合算进"排除周一"的因子均值，
   得出方向性错误的结论（PART13 §7）。
7. **噪声护栏。** 随机抽掉 1/3 交易日，ΔSharpe 的标准差是 0.15~0.80。
   所以 |ΔSharpe| < 0.15 一律当噪声，不管 p 值多好看。

## 5. 已知局限

* **QQQ 腿的参数与评价窗口有重叠**（见 §1），平静期需要最高 1.5 倍杠杆；收益主要来自 2020 年崩盘。
* 账户买力限制了资金规模：共用基数 $500k 时最坏同时持仓占买力 71%，$1M 时装不下。

* α 的 HAC t 值全样本 2.9，"可辩护"而非"高置信"。
* "排除周一"这个因子 dev≈0、test 显著、**全样本配对自助 p=0.23**，
  且 test 的收益集中在 3~5 个周一上。保留它是因为周一期望收益≈0
  （剔除近乎免费），**不应把它的 test 加成计入预期**（PART13 §7）。
* 模拟盘实盘验证自 2026-10-05 起，样本仍短。
* 收益高度非正态：偏度 +2.73、超额峰度 19.87（右偏厚尾），Sharpe 低估尾部特征。
* `audit_factors.py` 需要 `data/xs/` 横截面数据（108MB，未入库）。
* A 股横截面与美股 5 分钟数据受 moomoo 的 300 标的/7 天额度限制，未跑完。
* 微观结构（真实价差、OFI）**尚未实测** —— `scripts/micro_recorder.py` 没部署成。


## 2026-10 分层研究(新增,保留全部旧记录)
| 文档 | 内容 |
|---|---|
| `docs/ARCHITECTURE_layers.md` | 分层架构(L0 择时/regime、L1 RV、L2 Sleeve、L3 日内回转、L4 信息层)+ **证据地图与叠加表** |
| `docs/PROPOSAL_regime_layer.md` | L0 宏观/市场环境 regime 择时(两轮,42 配置;信用趋势 A2_C 为弱证据) |
| `docs/PROPOSAL_sleeve_anatomy.md` · `PROPOSAL_trend_rv_sleeve.md` | Sleeve 解剖;RV[0,1]/[0,1.5]/[0,2] 与趋势叠加 Sleeve |
| `docs/PROPOSAL_lowvol_avoidance.md` · `PROPOSAL_trade_filter.md` | 避开事后低波日(三轮)与逐笔入场过滤 |
| `docs/PROPOSAL_intraday_reversion_v2.md` | 分钟线日内回转(两轮,含文献 H1–H5) |
| `docs/PROPOSAL_llm_news_vol.md` | LLM 新闻→波动(三轮,含结构化事件抽取) |

**追加(2026-10-11,第三批)**:`docs/PROPOSAL_intraday_tech_indicators.md`(32 个经典技术指标 × QQQ 5/15 分钟,256 配置,无证据)· `docs/PROPOSAL_regime_layer.md` 第三轮(23 子信号二值库 + 行业龙头/落后 + EW/DYN 赋权,累计 60 配置,无证据)。脚本 `studies/tech_indicators.py`、`intraday_tech_study.py`、`regime_round3_study.py`。

**追加(2026-10-12,第四批)**:`docs/PROPOSAL_intraday_t_overlay.md`(事件驱动做T 倒T/正T,64 配置,无证据)· `docs/PROPOSAL_news_multisource.md`(多源新闻 → 日内波动,2019–2023,无证据;LLM 阶段待密钥)。脚本 `studies/tech_overlay_study.py`、`news_ms.py`、`lowvol_round4_news_study.py`、`news_ms_posthoc.py`。数据缓存 `ETFT_DATA/raw/news_ms/`(不入库)。

**追加(2026-10-12,第五批)**:`docs/PROPOSAL_vol_monetization.md`(波动模型能否变现:无)· `docs/PROPOSAL_fomc_day.md`(FOMC 日与时点偏移:线索)· `docs/PROPOSAL_credit_deepdive.md`(A2_C 细究 + 1986–2003 独立样本:弱,风险保险)。脚本 `studies/vol_monetization_study.py`、`vol_monet_m1b.py`、`fomc_day_study.py`、`credit_deepdive_study.py`。FRED 现已可达(缓存 `ETFT_DATA/raw/fred/`),长历史代理 `raw/regime/long_history.parquet`。

**追加(2026-10-12,第六批)**:`docs/PROPOSAL_disp_prediction.md`(位移/趋势效率预测 + 分阶段入场 + 分布面板:无证据)· `docs/PROPOSAL_iv_deepdive.md`(与隐含波动对赌:未达可交易)· `docs/PROPOSAL_macro_events.md`(CPI/NFP/零售/ISM + 新闻方向:无证据,CPI 缺口回补为事后线索)。脚本 `studies/disp_pred_study.py`、`iv_deepdive_study.py`、`macro_events_study.py`。数据缓存 `ETFT_DATA/raw/econ_cal/`(公开 GitHub 经济日历,含实际/预期/前值)。

**追加(2026-10-12,第七批)**:`docs/PROPOSAL_ffa_vol.md`(FFA 融合滤波 + 6 个基准:统计/仓位/避开低振幅日/择时)· `docs/PROPOSAL_event_delay.md`(事件日不入场/推迟全表 + CPI 回补 overlay 利润,事后)。脚本 `studies/ffa_vol_study.py`、`ffa_skip_posthoc.py`、`event_delay_study.py`。

**追加(2026-10-12,第八批)**:`docs/PROPOSAL_sota_stack.md`(避开低振幅日 13 模型赛马、反向仓位倾斜、Sleeve SOTA 组装、B&H 腿 78 配置跨周期 + 2019+ 叠加 Sleeve)。脚本 `studies/sota_stack_study.py`、`sota_posthoc.py`。

**追加(2026-10-12,第九批)**:新 Sleeve(FOMC≥14:30 + CPI 回补 + FFA-P 跳过 q=0.2)vs 实盘的 dev/test/全周期 × B&H 腿对比,见 `docs/PROPOSAL_sota_stack.md` 末节;脚本 `studies/sleeve_v2_compare.py`。

**追加(2026-10-12,第十批)**:`studies/tearsheet_v2.py` → `outputs/figs/tearsheet_v2.png`(上一版 4 条线 + 新 Sleeve + ① RV21+SMA200+新Sleeve + ② VT[FFA-SD]+SMA200+新Sleeve),指标表 `outputs/tables/tearsheet_v2_{metrics,periods}.csv`。`docs/FORWARD_LIVE_v2.md` = 前向部署与观察规格(分级、精确规则、日志字段、冻结的否决标准)。周一一致性修正见 `docs/PROPOSAL_sota_stack.md` 偏离记录 4。
