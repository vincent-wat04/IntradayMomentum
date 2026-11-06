# studies/ index

190 frozen research scripts, flat on purpose: they import each other as top-level modules (`import tearsheet as TS`), so **do not move or rename them**. Each script's header holds its preregistration. New work should use `quantkit/` (repo root) and add new scripts here only for study-specific logic.

Sub-packages: `etft/` (minute-bar loader and noise-area engine), `audit/` (mechanical tests, independent re-implementation, overfit audit, review checks), `data_events/` (event data).

Run from `studies/` with `export ETFT_DATA=<data root>`; use the repo venv. First line of each module docstring is shown below (auto-generated; some are in Chinese).

## canonical pipeline (reproduces v3/v4/v5)

| script | what it is |
|---|---|
| `b0_econ.py` | B0 经济检验(描述性):把 walk-forward 预测接到逐笔表上做"跳过最差 q% 笔"和"连续缩放", |
| `b0_entry_features.py` | B0:逐笔入场的条件期望 —— 新特征的构造与增量检验(描述性,不选配置)。 |
| `b0_eval.py` | B0 评估:新特征族相对 F1 的增量预测力(描述性,不选配置)。 |
| `b0_incremental.py` | B0 增量检验:新特征族相对 F1 的 walk-forward 增量 IC(描述性)。 |
| `b0_placebo.py` | B0 重训练安慰剂:DUR 特征块是否真的比"同分布的假 DUR"多提供信息? |
| `b0_validate.py` | B0 验证:对 b0_econ 里看起来最好的 DUR 格子做稳健性 + 家族级置换安慰剂。 |
| `cjkfont.py` | 中文字体解析：沙箱会定期清掉 /usr/share/fonts，所以必须自己兜底。 |
| `disp_pred_study.py` | 预测归一化位移 / 趋势效率 / 极低位移概率 + 分阶段入场 —— docs/PROPOSAL_disp_prediction.md(预注册)。python3 disp_pred_study.py |
| `duration_events.py` | 价格持续时间(price duration)事件时钟 —— 1 分钟 close 上的 PDS(price duration sampling)。 |
| `event_calendar.py` | 确定性事件日历(事前已知)。仅由日历规则决定,不使用任何价格。 |
| `ivol_objective_study.py` | A 部分:仓位模型的风险度量(docs/PROPOSAL_intraday_vol_entry.md §1,预注册)。 |
| `letf_posthoc.py` | 事后(看到 H1–H3 后追加,见 PROPOSAL_letf_rebalance.md 偏离记录): |
| `liquidity_inverse_study.py` | ① 流动性/换手(成本真实性、容量、作为信号) ② 买 SQQQ vs 空 TQQQ + 恐慌时仓位不对称 |
| `lowvol_avoidance_study.py` | 避开事后低波日 —— 严格按 docs/PROPOSAL_lowvol_avoidance.md 执行。 |
| `noise_core_dd.py` | 日频信号去噪、核心仓回撤控制、Sleeve : 核心仓配比(docs/PROPOSAL_noise_core_dd.md,预注册)。 |
| `paths.py` | 数据与引擎路径的唯一出口。所有脚本从这里取路径，不再写死本机目录。 |
| `r2_rules_study.py` | 第二轮 C 部分(规则进 SOTA v3)+ A 部分(仓位模型)—— docs/PROPOSAL_intraday_vol_entry_r2.md §1–2(预注册)。 |
| `r3_study.py` | 第三轮(docs/PROPOSAL_intraday_vol_entry_r3.md,预注册):入场带自适应、O2 迭代、替换 v3 中重叠的组件、新方法版 v3 的 tearsheet。 |
| `round3_common.py` | 第三轮预注册假设（H1 隐含-已实现波动缺口 / H2 FOMC 日 / H3 隐含相关性）共用的数据构造。 |
| `sizing_models_study.py` | 仓位模型的函数形式、复杂度与特征 —— 预注册(运行前写定,2026-10-22) |
| `sizing_sigma_feed.py` | σ14 直接喂给仓位模型 vs 在 σ14 波动目标之上叠加模型 —— 预注册(运行前写定,2026-10-20) |
| `sleeve_anatomy.py` | Sleeve 解剖 —— 严格按 docs/PROPOSAL_sleeve_anatomy.md 执行。 |
| `sota_posthoc.py` | 事后补充:SOTA 的『跳过』组件在 test 失败(dev 选 q=0.3)后,报告不含跳过的版本与 q=0.2 版本;以及推荐 B&H 腿 + Sleeve 版本的完整面板。 |
| `sota_stack_study.py` | 避开低振幅日的模型赛马 / 反向仓位倾斜 / Sleeve SOTA 组装 / B&H 腿参数(跨周期)—— docs/PROPOSAL_sota_stack.md(预注册)。 |
| `sota_tqqq.py` | 论文 SOTA 配置（vol targeting 4x + volume 调整 + GEX 调整）在 TQQQ 上的正确测试。 |
| `sota_unified_stack.py` | ① 统一仓位模型(U3p-clean)叠加到 SOTA Sleeve / 组合上(描述性)。docs/PROPOSAL_letf_rebalance.md §①。 |
| `tearsheet.py` | 策略 tearsheet：净值 / 回撤 / 月度 / 滚动 Sharpe / 完整指标表。 |
| `tearsheet_v2.py` | Tearsheet v2:在 tearsheet.py(上一版)的 4 条线基础上,叠加 |
| `tearsheet_v3.py` | Tearsheet v3:①/② 的 Sleeve 换成 v3 = SOTA(FOMC≥14:30 + CPI 回补 1× + FFA-P 跳过)+ 统一仓位(按各自组合重训)+ 15:40 平仓。 |
| `tearsheet_v3_variants.py` | v3 ① 组合(QQQ 腿 RV21×SMA200 + v3 Sleeve)的两个仓位模型变体回测(描述性,事后配置,不改变 v3 判定): |
| `tearsheet_v4.py` | Tearsheet v4:在 tearsheet_v3 的基础上,Sleeve 换成 v4 = K1。 |
| `tearsheet_v5.py` | Tearsheet v5 = v4 + (1) 买力上限 (2) 核心仓不交易带 (3) 等风险换配比 ρ = 1.5(固定)。定义在运行前写定(docs/PROPOSAL_noise_core_dd.md §6)。 |
| `unified_sizing_posthoc.py` | 事后(看到 U3 结果后追加,见 PROPOSAL_unified_sizing.md 偏离记录): |
| `unified_sizing_study.py` | 统一的仓位模型:低波动日缩仓 + 恐慌日空头加码放进同一个可学习的 sizing 框架; |

## review / audit / validation (2026-10)

| script | what it is |
|---|---|
| `hf_exec_cost.py` | 用 mdrec 的 TQQQ 10 档盘口检查回测成本假设(半价差 $0.005/股)在订单变大时是否还成立。 |
| `hf_mdrec.py` | mdrec 逐笔 + 10 档盘口 → 每分钟特征(按美东交易日分组;只取常规时段 09:30–16:00)。 |
| `hf_validate.py` | 用 mdrec 真实逐笔/盘口(2026-10-20 ~ 10-30,5 个完整交易日 + 1 个半天)验证此前研究用的 1 分钟代理与成本假设。 |
| `noise_band_validation.py` | Statistical validation of the paper and current adaptive Noise Band designs. |
| `r3_ffa_check.py` | 细查:去掉 FFA-P 跳过为什么 Sharpe 更高 —— 是否因为仓位模型(尤其线性)本来就在这些日子给很小的仓位?(2026-10-23,事后诊断;新配置 +1:ffa_rank 连续特征) |
| `r3_ffa_learn.py` | 让仓位模型自己学 FFA 跳过(2026-10-23,事后;新配置 +5)。 |
| `r3_followup.py` | 第三轮事后追加 2(2026-10-23;累计 +4): |
| `r3_penalty.py` | 第三轮事后追加 3(2026-10-23;累计 +2):固定 E1 入场带 + CLEAN+dur_rate + 线性链接,只换风险惩罚: |
| `r3_stack.py` | 第三轮事后追加(看到 r3_study 结果后,2026-10-23;累计 +4):三个最接近的候选叠加 + "要不要去掉 FFA-P 跳过"的专门检验。 |

## crowding & noise/core study

| script | what it is |
|---|---|
| `crowding_daily.py` | 日更拥挤度(docs/PROPOSAL_crowding_daily.md §1)。 |
| `crowding_daily_strategy.py` | 日更拥挤度 → 策略(docs/PROPOSAL_crowding_daily.md §2,预注册):H1–H6,相对 v4 ①。 |
| `crowding_daily_validate.py` | H5 的验证(看到结果后追加,诊断;新配置 +2:V1a、V1b)。 |
| `crowding_followup.py` | 拥挤度追问(2026-10-29): |
| `crowding_index.py` | 收益隐含拥挤度(Feng 2027,按微信解读还原)—— docs/PROPOSAL_crowding_vol_timing.md §2。 |
| `crowding_study.py` | 拥挤度 × Sleeve:关系检验(B1–B5)与纳入策略(C1–C6、D1)—— docs/PROPOSAL_crowding_vol_timing.md §3–4(预注册)。 |
| `noise_core_bp.py` | 买力约束下的配比(事后,2026-10-29;新配置 +2)。 |

## intraday research (reversion, indicators, T-overlay, stops, take-profit, GEX)

| script | what it is |
|---|---|
| `cojump_cos.py` | OHLC-COS 共同跳跃检测器(按微信文章文字重建,**不是**论文代码)。 |
| `cojump_cos_feasibility.py` | OHLC-COS 共同跳跃检测器:可行性与校准(**不使用任何收益/策略数据,不选参**)。 |
| `convexity_strategy.py` | 基于"收益形态是凸性"这个发现去开发策略。 |
| `event_delay_study.py` | 事件日:不入场 / 推迟入场 是否普遍更好?(对所有事件与结果分类)+ CPI 缺口回补 overlay 能加多少利润。 |
| `event_sleeve_study.py` | 事件日 × Sleeve(探索性,第二轮 E 组的直接检验;日历事前已知,无拟合参数)。 |
| `fomc_day_study.py` | FOMC 决议日怎么交易最好 / 交易时点要不要偏移 —— docs/PROPOSAL_fomc_day.md(预注册)。python3 fomc_day_study.py |
| `gamma_proxy_study.py` | 做市商 gamma：论文用 RSI(5) 做代理，我们手上有真实 GEX —— 谁在带信息？ |
| `gex_decay_study.py` | GEX 对当日波动幅度的预测力,在 0DTE 时代(2022-05 每日到期上线)是否衰减 —— proposal_sleeve_anatomy H2。 |
| `gex_false_breakout.py` | GEX 能不能过滤假突破？—— 这次按正确的假设去测。 |
| `gex_lag_study.py` | GEX 的滞后代价：t−2（我用的）vs t−1（合法最小）vs t−0（前视，只做上界）。 |
| `gex_reversion_study.py` | GEX 的另一面：高 gamma 日跑**反向**策略，以及尾盘反转。 |
| `gex_scaling_raw.py` | 用**原始 GEX**（而非分位）做仓位缩放 + 配对区块自助检验。 |
| `gex_skip_study.py` | 规避极高 GEX 日是否有 edge —— 严格按 docs/PROPOSAL_gex_skip.md 执行。 |
| `gex_vol_forecast.py` | GEX 用于**波动率预测**，再传导到 vol targeting 的分母。 |
| `grid_phase_ensemble.py` | 检查时点网格的相位集合：审计暴露出来的优化线索。 |
| `intraday_battery.py` | 日内策略全家桶：TQQQ 版日内动量、隔夜仓的影响、以及 8 种日内回转策略的对照实验。 |
| `intraday_cross_section.py` | 日内横截面策略：隔夜跳空反转、首半小时动量，全部**不持隔夜仓**（收盘 MOC 平仓）。 |
| `intraday_factor_library.py` | Minute/5-min bar aggregated daily factors (realized-measure family). |
| `intraday_reversion_v2_study.py` | 日内回转 v2 —— 严格按 docs/PROPOSAL_intraday_reversion_v2.md 执行。 |
| `intraday_reversion_v2b_study.py` | 日内回转 v2 第二轮 —— docs/PROPOSAL_intraday_reversion_v2.md「第二轮」。H1 周期性 / H2 末半小时 / H3 BVC 失衡 / H4 缺口+趋势 / H5 ML v2.1。 |
| `intraday_tech_study.py` | 32 个经典技术指标在 QQQ 5/15 分钟 K 线上的日内检验 —— docs/PROPOSAL_intraday_tech_indicators.md(预注册)。 |
| `inverse_etf_study.py` | 用反向 ETF 实现空头腿：SQQQ(-3x) / QID(-2x) / PSQ(-1x) vs 直接做空 TQQQ。 |
| `long_short_decomposition.py` | 多空双向 vs 只做多：空头腿到底贡献了什么。 |
| `overwrite_study.py` | 对 QQQ 腿卖月度备兑看涨：合成定价回测。严格按 docs/PROPOSAL_overwrite.md。 |
| `short_leg_diagnosis.py` | 空头胜率为什么低（35.6% vs 多头 46.3%）？用成交明细逐笔拆开看。 |
| `sleeve_v2_compare.py` | 新 Sleeve(实盘 + FOMC≥14:30 + CPI 回补 1× + 跳过 FFA-P 预测最低 20% 振幅日)vs 实盘 Sleeve: |
| `stop_decision.py` | 止损最终选型：vwap（当前）vs band+硬尾0.5σ。以及周一那 5 天到底说明了什么。 |
| `stop_design_study.py` | 止损设计空间的系统扫描：有没有一种只切左尾、不切正常回撤的止损？ |
| `stop_hard_tail.py` | 逐根 bar 连续评估的硬尾部止损 —— 「只切极端左尾」的正确实现。 |
| `stop_removal_study.py` | 去掉止损（尤其是空头止损）的收益与风险分析。 |
| `stop_significance.py` | 止损改动到底显著吗：配对区块自助法（paired block bootstrap）。 |
| `takeprofit_diag.py` | 止盈的前置诊断：到底有没有"逃顶"这回事？ |
| `takeprofit_gex.py` | GEX 能不能告诉我们"今天该不该止盈"？ |
| `takeprofit_placebo.py` | 止盈的收益到底来自"择时"还是"少做空"？—— 安慰剂对照 + 叠加层口径。 |
| `takeprofit_study.py` | 止盈 × 同向多次开仓：策略层实测。 |

## daily timing / regime / vol-forecast research

| script | what it is |
|---|---|
| `alpha_beta_decomp.py` | 12 个版本的 alpha / beta 分解：到底是 alpha 还是只是市场敞口？ |
| `credit_deepdive_study.py` | A2_C 类(信用趋势)特征的细致研究 —— docs/PROPOSAL_credit_deepdive.md(预注册)。python3 credit_deepdive_study.py |
| `extended_timing_research.py` | Point-in-time intraday-volume and price proxies for SPY/QQQ timing. |
| `fear_derisk_study.py` | 恐慌族因子能否降低组合回撤 —— 按 docs/PROPOSAL_fear_derisk.md 执行。 |
| `ffa_skip_posthoc.py` | 事后(探索性):E2『避开预测的低振幅日』的稳健性 —— 邻近阈值、置换族、组合层、振幅 vs RV 的差别。 |
| `ffa_vol_study.py` | FFA(融合滤波)与 6 个基准在 QQQ 上:统计、分状态、Sleeve 仓位 / 避开低波日 / 波动择时 —— docs/PROPOSAL_ffa_vol.md(预注册)。 |
| `improvement_tests.py` | 四类改进的检验：不对称止损、波动乘数 VM、论文 §4.2 日内形态、§4.3 星期效应。 |
| `improvement_tests2.py` | 改进检验第二轮：NR4 的置换检验、长期信号、fast alpha 执行层、微观结构确认。 |
| `improvement_tests3.py` | 改进检验第三轮：排除周一的置换检验、fast alpha 执行层、微观结构确认。 |
| `index_timing_framework.py` | Reusable SPY/QQQ noise-band factor event and portfolio backtest. |
| `iv_deepdive_study.py` | 与隐含波动对赌的深究:包含回归 + 0DTE 跨式代理 —— docs/PROPOSAL_iv_deepdive.md(预注册)。python3 iv_deepdive_study.py |
| `iv_rv_panic_study.py` | 隐含 vs 已实现波动的错位 + 恐慌后加仓 —— 按 docs/PROPOSAL_iv_rv_panic.md 执行。 |
| `leverage_frontier.py` | Stage 8: the return / drawdown frontier with real leveraged ETFs, and the corrected verdict. |
| `leveraged_portfolio.py` | Stage 6: turn the alpha sleeve into actual strategy return. |
| `lowvol_robustness.py` | lowvol_avoidance_study 的事后稳健性(探索性,不用于选配置):skip 分位邻域扫描 + 逐年 + 被跳过日的画像。 |
| `lowvol_round2_study.py` | lowvol_avoidance 第二轮:加入确定性事件日历 E 组 —— 见 docs/PROPOSAL_lowvol_avoidance.md「第二轮」。 |
| `lowvol_round3_study.py` | lowvol_avoidance 第三轮:已发表的新闻文本指数 N 组(GPR/EPU)。见 docs/PROPOSAL_lowvol_avoidance.md「第三轮」。 |
| `lowvol_round4_news_study.py` | 多源新闻 N 组 → 次日日内波动 / 低波标签(docs/PROPOSAL_news_multisource.md,第一阶段:非 LLM 特征)。 |
| `macro_events_study.py` | CPI / NFP / 零售 / ISM 日内检验与新闻方向 —— docs/PROPOSAL_macro_events.md(预注册)。python3 macro_events_study.py |
| `macro_factor_study.py` | Macro / cross-asset / sentiment factors from the etf_timing project, evaluated with THIS |
| `placebo_transform.py` | Null distribution for every (map, agg, h): REAL factors circularly shifted in time by a random offset |
| `portfolio_construction.py` | Stage 7: why the sleeve did not convert into return, and the construction that does convert it. |
| `raven_lite_timing.py` | Small-sample RAVEN-inspired adaptive-context diagnostic, NOT a paper replica. |
| `raven_paper_model.py` | Paper-architecture RAVEN for daily ETF factors (requires PyTorch). |
| `regime_data.py` | 宏观/市场环境数据(长历史)。全部为公开来源,缓存到 ETFT_DATA/raw/regime。 |
| `regime_features.py` | L0 宏观/市场环境特征 —— 见 docs/PROPOSAL_regime_layer.md §1–§2。所有序列在 t 日收盘形成,只用 ≤t 的数据。 |
| `regime_round2_study.py` | L0 Regime 第二轮 —— docs/PROPOSAL_regime_layer.md「第二轮」。复用第一轮构件。 |
| `regime_round3_study.py` | L0 Regime 第三轮 —— docs/PROPOSAL_regime_layer.md「第三轮」:二值子信号库(开源 14 美股类比)+ 行业龙头/落后 + EW/DYN 赋权。 |
| `regime_sleeve_study.py` | L0→L2:宏观 regime 状态下 Sleeve 的表现(描述性 + 圆周平移置换)。regime 信号取自 regime_round2(t−1 收盘的分数,t 日交易)。 |
| `regime_timing_study.py` | L0 宏观/市场环境 Regime 择时 —— 严格按 docs/PROPOSAL_regime_layer.md 执行。 |
| `sens_rv_weights.py` | RV21 降风险的权重参数敏感性分析。 |
| `slowbear_robust.py` | 慢熊研究的事后稳健性检查（**全部标记为事后探索**，不能改动 PROPOSAL 第 3 节的判定）。 |
| `slowbear_study.py` | 慢熊行情的处理 —— 按 docs/PROPOSAL_slowbear.md 冻结的规则运行。 |
| `slowbear_verdict.py` | 按 PROPOSAL_slowbear.md 第 3 节的冻结规则自动判定（读 outputs/slowbear_study.csv，不重新挑选）。 |
| `test_index_timing_framework.py` | Timing regression checks for the reusable factor backtest. |
| `timing_combination_study.py` | Stage 2: combining many time-series predictors into ONE timing signal (SPY / QQQ, h = 5, 20, next_open). |
| `timing_model_strategy.py` | Stage 3 (model) + Stage 4 (strategy): screened factors -> walk-forward models -> tradable strategies. |
| `timing_transform_report.py` | Figures + tables for timing_transform_study / placebo_transform / timing_combination_study. |
| `timing_transform_study.py` | How should a raw time-series predictor be turned into a timing position? (SPY / QQQ, 2019-01 .. 2026-10) |
| `trend_rv_sleeve_study.py` | QQQ 腿的 RV × 趋势(含 lo=0 / 空仓)与 Sleeve 叠加 —— 严格按 docs/PROPOSAL_trend_rv_sleeve.md 执行。 |
| `vol_monet_m1b.py` | M1 的诊断补充(事后,看到神谕缩放为负之后追加,见 PROPOSAL_vol_monetization.md 偏离记录): |
| `vol_monetization_study.py` | A+B+E 波动模型能否变现 —— docs/PROPOSAL_vol_monetization.md(预注册)。python3 vol_monetization_study.py |
| `vol_sizing_exponent.py` | 仓位 ∝ σ^β 的指数扫描：从 vol targeting（β=−1）一直扫到反向（β=+1）。 |

## cross-section / factors (incl. China A-share)

| script | what it is |
|---|---|
| `analyze_full.py` | Summarise the full-sample (2019-01..2026-10, test from 2023-01-03) factor and model studies. |
| `cn_adaptivity_study.py` | Stage 9: does the pipeline ADAPT, or are the "good factors" fixed? A-share vs US. |
| `cn_cross_section.py` | A 股中小盘横截面（中证500 + 中证1000 成分，剔除 ST），研报协议。 |
| `cn_strategy.py` | Stage 10: the same pipeline applied to A-share, with a null. |
| `combo_comparison.py` | 静态再配比 × RV 降风险 的组合业绩对比。 |
| `combo_lib.py` | QQQ 腿的 RV 降风险回测：tearsheet 与 combo_comparison 共用，避免各写一份。 |
| `consolidate_single_symbol.py` | Every strategy we have run, reported on SPY and on QQQ separately (no 50/50 blending). |
| `cross_section_models.py` | Stage 11b: do GBM / random-feature ridge beat a linear model once we have a cross-section? |
| `cross_section_study.py` | Stage 11: the cross-section. Does the extra data make the complex models work? |
| `factor_sharpe.py` | Single-factor timing Sharpe (SPY / QQQ, 2019-01 .. 2026-10). |
| `factor_sharpe_report.py` | Figures + tables for single-factor Sharpe (reads factor_sharpe.csv and cache/factor_books.pkl). |
| `final_strategy.py` | Stage 5: the final strategy, built from the pipeline chosen on dev evidence only, plus all figures. |
| `fundamental_factors.py` | Point-in-time fundamental factors from SEC XBRL, for the 98-name universe. |
| `multifrequency_models.py` | Causal horizon-specific selection and SPY/QQQ timing model comparison. |
| `pdf_factor_test.py` | Test the 东北证券 volume-distribution factors (pdf_volume_factors) where intraday data exists. |
| `pdf_volume_factors.py` | Intraday volume-distribution factors from 东北证券《日内成交量分布因子及 Logsig-Alpha 因子生成》 |
| `pdf_xs_replication.py` | 研报因子的真正横截面复现：用 moomoo 5 分钟 bar，不再用日频代理。 |
| `protocol_universe_compare.py` | Large vs mid vs small cap, under the exact 研报 protocol, plus factor correlation and style exposure. |
| `report_protocol.py` | 东北证券研报协议的严格复现 + 为什么我们的 Sharpe 比研报低（广度分解）+ 相关性与风格暴露。 |
| `run_full_extended.py` | Full-sample driver for extended_timing_research (2019-01 .. 2026-10 moomoo data). |
| `run_full_models.py` | Full-sample model study (2019-01 .. 2026-10): rolling Ridge / ElasticNet, RAVEN-lite and the |
| `xs_attribution.py` | Why is the single-factor Sharpe ~1.0 but the model composite ~0? A controlled ladder. |
| `xs_factor_library.py` | Cross-sectional factor library: 78 factors from daily OHLCV, including the microstructure family. |
| `xs_families.py` | Single-factor cross-sectional results for all 77 factors, broken out by family. |
| `xs_models.py` | Cross-sectional models on the 77-factor library: does complexity pay once N x T is large? |
| `xs_multifreq.py` | Fundamentals + how to combine a low-frequency signal with high-frequency ones. |
| `xs_pipeline_v2.py` | Cross-sectional pipeline v2 -- built to the setup you specified. |
| `xs_turnover.py` | Keep the style exposure, then attack turnover. Five reduction methods compared. |

## news / LLM / data fetch

| script | what it is |
|---|---|
| `gdelt_fetch.py` | GDELT DOC 2.0 日度新闻量/语调(聚合,无原文)。限流 1 请求/5 秒。输出 ETFT_DATA/raw/gdelt/*.parquet |
| `llm_event_extract.py` | LLM 逐条新闻结构化事件抽取 —— docs/PROPOSAL_llm_news_vol.md「第三轮」。 |
| `llm_news_ms.py` | 第二阶段:多源新闻头条 → LLM 结构化事件抽取 → 次日日内波动检验。docs/PROPOSAL_news_multisource.md「第二阶段补充登记」。 |
| `llm_news_vol_r2.py` | LLM 新闻→波动 第二轮 —— 见 docs/PROPOSAL_llm_news_vol.md「第二轮」。 |
| `llm_news_vol_study.py` | LLM(新闻 ± 宏观上下文)预测 10:00 之后波动 —— 严格按 docs/PROPOSAL_llm_news_vol.md 执行。 |
| `news_ms.py` | 多源新闻语料(HF bucket 缓存)→ 决策日 D 的新闻特征。见 docs/PROPOSAL_news_multisource.md。 |
| `news_ms_posthoc.py` | 事后(探索性)检查:第一阶段单变量里过 Bonferroni 的 w_mk_GEO / w_mk_BANK 是否真有预测增量。 |
| `write_model_report.py` | Generate auditable Markdown and CSV from the frozen six-horizon model outputs. |

## plots / reports / misc

| script | what it is |
|---|---|
| `_repro_headline.py` | Reproduce the README headline row (多空 band+尾0.5 无周一) by REUSING strategy_comparison.main() internals. |
| `bear_regime_diag.py` | 熊市里 RV 降风险到底在做什么：红圆 lo=0 vs 浅蓝 lo=0.5。 |
| `ivol_entry_study.py` | B 部分:日内波动时间分布特征(IVT)→ 入场筛选(docs/PROPOSAL_intraday_vol_entry.md §2,预注册)。 |
| `ivol_posthoc.py` | 事后(看到 ivol_objective / ivol_entry 结果后追加;不改变预注册判定;累计 +1): |
| `layer_stack_study.py` | 分层叠加表(ARCHITECTURE_layers.md):逐层累加已登记的配置,报告每层相对前一层的增量。不做新的选择。 |
| `letf_rebalance_study.py` | ② 杠杆 ETF 收盘再平衡流:增厚利润(H2)还是优化执行(H3/H4)?—— docs/PROPOSAL_letf_rebalance.md(预注册)。 |
| `plot_alpha_beta.py` | alpha / beta 分解图。 |
| `plot_comparison.py` | 12 个版本的完整对照图。 |
| `plot_sens_rv.py` | RV21 降风险权重敏感性：热力图（面板 A 的 ΔSharpe 与 ΔMDD）。 |
| `plot_slowbear.py` | 慢熊研究配图：回撤曲线（SPX / NDX，1998-2010 与 2020-2026）。 |
| `plot_systematic.py` | 系统化研究配图：关闭态权重 e_off 的影响（7 个市场均值 + 个别市场）。读 systematic_headline.csv。 |
| `plot_takeprofit.py` | 止盈研究的四张图。 |
| `qqq_overlay_check.py` | Does the stage-5 "best" strategy (B&H + risk-budgeted overlay, GP partial trading) beat QQQ alone? |
| `r2_factor_zoo.py` | 第二轮 B 部分:高频因子库(HFZ 10 个)+ 跨资产(XA 4 个)→ 逐笔入场筛选(docs/PROPOSAL_intraday_vol_entry_r2.md §3,预注册)。 |
| `r2_zoo_in_v3.py` | 第二轮 §5:把 r2_factor_zoo 中满足 Sleeve 层判定的 3 格(D1/D2/D3)带入 v3 组合复测。 |
| `reconcile_numbers.py` | 把所有口径的 Sharpe 对齐成一张账：到底哪个数字是什么意思。 |
| `round3_study.py` | 第三轮：H1/H2/H3 主检验（Bonferroni）+ RV21 体制三检验 + 象限分解。 |
| `sizing_durrate_followup.py` | 事后追加(2026-10-22,看到 sizing_g5_followup 之后): |
| `sizing_g5_followup.py` | 事后追加(看到 sizing_models_study 结果后,2026-10-22):F1 线性链接 + G5 价格持续期特征;G5 的特征重要性。 |
| `strategy_comparison.py` | 几个候选版本的完整对照：PnL 曲线 + Sharpe / 年化 / Calmar / 分边胜率与盈亏比。 |
| `systematic_trend_rv.py` | RV × 趋势 的系统化组合 + 参数证据 + 做空 —— 按 docs/PROPOSAL_systematic.md 冻结的规则。 |
| `tech_indicators.py` | 32 个经典技术指标(东北证券《量化择时系列一》)—— numpy/pandas 手写(TA-Lib 不可用)。 |
| `tech_overlay_study.py` | 日内『做T』叠加仓(倒T/正T,事件驱动)—— docs/PROPOSAL_intraday_t_overlay.md(预注册)。 |
| `test_extended_research.py` | Focused checks for new factor aggregation and model execution. |
| `test_multifrequency_models.py` | Causality and architectural smoke checks for the new project. |
| `trade_filter_study.py` | Sleeve 逐笔入场过滤 —— 严格按 docs/PROPOSAL_trade_filter.md 执行。 |
