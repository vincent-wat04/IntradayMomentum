# PROPOSAL — 仓位模型的风险度量 + 日内波动时间分布的入场特征(预注册,2026-10-22)

分支 `research/intraday-vol-entry-2026-10`(基于 `research/duration-vol-entry-2026-10` bb5eff5)。数据口径同 `REPORT_unified_sizing.md` §8(分钟线 md5 = 服务器;其余公开源重建,清单 `data/MANIFEST_sandbox_2026-10.md5`)。
本文在运行任何新实验之前写定;结果出来后不改配置,只追加结果与偏离记录。

## 0. 现状(从代码读出,不是推断)
* 仓位模型目标(`unified_sizing_posthoc.ppp_fit`):`U = mean(R) − γ/2 · var(R) − λ‖θ‖²`,`R_t = k_l·L_t + k_s·S_t + E_t`(bp/天;L、S = Sleeve 多/空腿日损益,E = QQQ 腿 + CPI overlay),`γ = mean(B)/var(B)`(B = k≡1 的训练期组合)。
* 也就是说:**唯一的"波动率估计"是训练期日损益的等权样本方差**(跨日、无条件)。目标里**没有任何日内波动**,也不直接含位移;位移只通过 `mean(R)` 间接被奖励。
* 两个问题(先验):① 对右偏的 Sleeve,`var(R)` 把大位移带来的**上行**也当风险惩罚;② 单日 `(R_t − R̄)²` 是当天风险的高噪声估计,日内已实现方差(分钟收益平方和)是更有效的估计(Andersen–Bollerslev–Diebold–Labys 2003)。

## 1. A 部分:风险度量(仓位模型)
诊断(描述性,不选择):
* D1 Sleeve(SOTA、未做仓位缩放)日损益对 `log dn`(开盘到收盘位移 / σ14)与 `log nrv`(日内 1 分钟 RV / σ14²)联合回归(HAC t),分多空腿 —— 回答"收益是否由位移驱动、给定位移后日内波动是否有害"。
* D2 现行模型 OOS 的 `log k` 与事后 `log dn`、`log nrv` 的相关 —— 回答"现在的模型实际在奖励/惩罚什么"。
配置(2 个,其余与 B 完全相同:sigmoid、CLEAN 7 特征、λ 网格、留一年 CV、OOS 2021–26;CV 用各自的目标函数打分):
* **O2 下行半方差**:`U = mean(R) − γ₂/2 · mean(min(R,0)²)`,`γ₂ = mean(B)/mean(min(B,0)²)`。
* **O4 日内已实现方差**:`U = mean(R) − γ₄/2 · mean(QV_t(k))`,`QV_t = Σ_分钟 (组合分钟损益)²`(含隔夜缺口项与腿间交叉项,Sleeve 空头对 QQQ 腿的日内对冲被计入),`γ₄ = mean(B)/mean(QV_t(1))`。分钟损益由 30 相位成交明细逐分钟重建(capital=1,股数即 NAV 单位),须与腿的毛损益对账。
判定(同上一轮):组合 OOS ΔSR > 0、dev(21–22)/test(23–26) 同正、块自助 95% 下界 > −0.05;另报组合日内已实现波动的变化、分布面板。

## 2. B 部分:日内波动时间分布特征 → 入场筛选(逐笔)
样本:`_TR.pkl` 的 29,679 笔(实盘 Sleeve 30 相位;与 B0 同一框架,组合层 = L1 RV[0.5,1.5] + Sleeve)。
**新特征族 IVT**(决策 bar k 及之前,PIT;方向性特征乘以交易方向):
| 特征 | 定义 | 依据 |
|---|---|---|
| `vr5`, `vr10` | 截至 k 的 5/10 分钟收益 RV ÷ 1 分钟 RV(方差比;>1 = 趋势性路径,<1 = 来回噪声) | Lo–MacKinlay 方差比;volatility signature |
| `rv_surp` | log(截至 k 的累计 RV ÷ 过去 20 日同一时点累计 RV 均值) | 日内周期性(Andersen–Bollerslev 1997) |
| `rv_com_dev` | RV 时间重心(Σ m·r²/(k·Σ r²))减去过去 20 日同一时点均值 | 波动集中在早段还是近段 |
| `rv_ent` | 5 分钟桶 RV 份额的归一化熵(低 = 波动集中在少数突发) | 波动的时间集中度 |
| `jump_k` | 1 − 双幂变差/RV(截至 k) | Barndorff-Nielsen–Shephard |
| `rs_sgn` | 方向 ×(RS⁺ − RS⁻)/RV(带符号已实现半方差) | Barndorff-Nielsen–Kinnebrock–Shephard 2010;Patton–Sheppard 2015 |
| `rskew_sgn` | 方向 × 已实现偏度 | Amaya et al. 2015 |
| `eff_k` | \|log(c_k/O)\| / Σ\|r\|(路径效率) | Kaufman 效率比 |
**目标(两个,预先固定)**:`y` = 单笔净回报(bp,现有);`y_ra` = `y / (1e4·√RV_hold)`,RV_hold = 持仓期间 1 分钟对数收益平方和(风险调整后回报,"单笔的已实现夏普"),截断到 [−10, 10]。
自检:未来投毒(k 之后的价格 ×7.3,特征逐位不变)、阳性对照(入场后 5 分钟收益)。
检验:
1. 单变量 Spearman IC:9 特征 × 2 目标 × {2019–22, 2023–26}(描述性)。
2. 增量:walk-forward(2021–26,逐年重估;ridge/lgbm 与 `trade_filter_study` 冻结超参相同)F1 vs F1+IVT 的 ΔIC,按交易日聚类自助。
3. 经济(4 个配置 = {ridge, lgbm} × {y, y_ra},特征 F1+IVT):跳过因果分位最差 20%(与 B0、trade_filter 相同的 q,不扫描),Sleeve 层与组合层 ΔSR;参照线 = 同模型同目标的 F1-only。
   族内校正:年内逐笔置换 500 次,4 格取 max。
判定:"有证据" = Sleeve ΔSR dev(21–22) 与 test(23–26) 同正、全 OOS ≥ +0.10、族内 p < 0.05,**且**高于对应 F1-only 线;否则为"线索"或"无"。
配置计数:A 2 + B 4 = 6(累计在 `REPORT_unified_sizing.md` §8 基础上 +6)。

## 偏离记录
(运行后追加)

## 参考
* Andersen & Bollerslev (1997), *Intraday periodicity and volatility persistence in financial markets*, J. Empirical Finance —— [PDF](https://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/c9d1a098610563d0c12581f60072a23a/$FILE/AndersenBollerslev1997b.pdf)
* Patton & Sheppard (2015), *Good volatility, bad volatility*, REStat —— [PDF](https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf);Bollerslev (2022) 综述 [PDF](https://public.econ.duke.edu/~boller/Papers/Semi_JFEC_2022.pdf)
* Gao, Han, Li & Zhou (2018), *Market intraday momentum*, JFE —— [PDF](https://c.mql5.com/forextsd/forum/173/intraday_momentum_-_the_first_half-hour_return_predicts_the_last_half-hour_return.pdf);Baltussen et al. (2021) 归因于 gamma 对冲
* Barndorff-Nielsen, Kinnebrock & Shephard (2010) 已实现半方差;Amaya, Christoffersen, Jacobs & Vasquez (2015) 已实现偏度;Lo & MacKinlay (1988) 方差比
