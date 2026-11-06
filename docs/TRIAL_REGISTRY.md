# 试验登记簿(Trial Registry)— 全项目多重检验记账

目的:handbook 第 2 讲 / Bailey & López de Prado (2014) 的 Deflated Sharpe 要求知道"一共试了多少次"。此前各研究在自己的文档里各记 "+N",没有项目级总账。本表是唯一权威计数,新研究必须在此追加一行(`studies/audit/overfit_audit.py` 读取"合计"行)。
口径:一行 = 一轮研究;"配置" = 产生了一条可比较的组合/Sleeve 收益序列或一个经济检验格子;描述性统计表(单变量 IC、相关矩阵)不计入,但在"描述性检验"列另记。数字取自各文档的计数段落(2026-10-31 汇总)。

| # | 研究轮 | 文档 | 层 | 配置 | 描述性检验 | 预注册 | 结论 |
|---|---|---|---|---|---|---|---|
| 1 | 早期分层研究(L0 宏观/信用/多信号、L3 回转、技术指标、做T、新闻、低波避开等) | `ARCHITECTURE_layers.md` 汇总 | L0–L4 | ≈150(文档自述"~150",未逐项核对) | — | 部分 | 仅 Sleeve 显著 |
| 2 | SOTA 组装、FFA、事件日、统一仓位、LETF(v2/v3 的来源) | `PROPOSAL_sota_stack.md` 等 | L1/L2 | ≈60(**估计**:依据各文档零散的"42→60""50 个配置"等计数,未逐项核对) | — | 部分 | v3 组件均为事后选择 |
| 3 | B0 价格持续期 / 微观 / 分段 RV / VOLO | `PROPOSAL_duration_vol_entry.md` §11 | L2 入场 | 70 | 81 + 增量 IC | 否(事后) | DUR 线索 |
| 4 | σ14 喂给仓位模型 | `REPORT_unified_sizing.md` §7 | L2 仓位 | 3 | — | 是 | 无 |
| 5 | 链接函数 / 复杂模型 / RAVEN / 特征族 | `REPORT_unified_sizing.md` §8 | L2 仓位 | 11 | — | 是 | 无 |
| 6 | F1+G5、F1+dur_rate 等事后 | `REPORT_unified_sizing.md` §8.1–8.2 | L2 仓位 | 3 | 重要性 | 否 | 线索 |
| 7 | v3 变体 tearsheet(线性、线性+dur_rate) | `tearsheet_v3_variants.py` | L2 仓位 | 2 | — | 否 | 描述 |
| 8 | 日内波动第一轮(目标函数 O2/O4、IVT 入场) | `PROPOSAL_intraday_vol_entry.md` | L2 | 7 | 18 | 是 6 + 事后 1 | 线索 |
| 9 | 日内波动第二轮(rv_surp、带宽、HFZ、XA、v3 复测) | `PROPOSAL_intraday_vol_entry_r2.md` | L2 | 16 | 28 | 是 13 + 事后 3 | 无 |
| 10 | 第三轮(入场带、O2 迭代、替换、阶梯) | `PROPOSAL_intraday_vol_entry_r3.md` §5 | L2 | 15 | — | 是 | 无 |
| 11 | 第三轮事后(叠加、机制、惩罚、FFA 检查、FFA 学习) | 同上 §6–10 | L2 | 18 | — | 否 | v4 = K1 |
| 12 | 拥挤度(月频) | `PROPOSAL_crowding_vol_timing.md` | L0/L2 | 9 | B1–B5 | 是 7 + 事后 2 | 无 |
| 13 | 日更拥挤度 | `PROPOSAL_crowding_daily.md` | L0/L2 | 8 | — | 是 6 + 事后 2 | H5 通过 |
| 14 | 去噪 / 核心仓 / 配比 / 买力约束 | `PROPOSAL_noise_core_dd.md` | L1/组合 | 10 | 长样本 | 是 8 + 事后 2 | S2c 通过 |
| 15 | v5 + 敏感性 | `tearsheet_v5.py` | 组合 | 4 | — | ρ 预先固定 | v5 |
| 16 | 研究链复盘检查(R2 禁运 3、R3a 简化跳过 4、R3b dur_rate 2、R3d k≡1 1) | `REVIEW_2026-10.md`、`studies/audit/review_checks.py` | L2 | 10 | R1/R4/R5 诊断(PSI 8、归因 2、单调性 8) | 是(判据写在脚本头) | 无新增策略;仅诊断 |
| | **合计** | | | **≈ 396** | | | |

注意:合计是下界(早期轮次的"≥150"本身就是下界,参数网格内部的每个点没有单独计数)。
