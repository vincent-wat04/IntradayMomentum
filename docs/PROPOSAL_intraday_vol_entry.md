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

## 3. 结果(2026-10-22)
### A 部分(`studies/ivol_objective_study.py`)
对账:分钟损益重建之和 = 腿毛损益(差 4e-17);QQQ 腿分钟之和 = 日收益(差 3e-12bp);B 与 repo `P.wf` 的 k 差 2.8e-15。
* **D1**(Sleeve 未缩放日损益,2019–26):`log dn` 系数 +45.5bp(t 15.0),`log nrv` −16.9bp(t −4.8),R² 0.23。**收益由位移驱动;给定位移,日内波动越高越亏**。多头腿对日内波动更敏感(t −5.9),空头腿不显著(t −1.3)。单看相关,Sleeve 与 nrv 几乎无关(0.03),因为 nrv 与 dn 正相关 0.38,两个效应互相抵消。
* **D2**(现行模型 OOS 的 log k 与事后变量 Spearman):多头 k 与 dn +0.12、与 nrv **+0.17**;空头 k 与 dn −0.14、与 nrv −0.33。现行模型在高日内波动的日子**加**多头、**减**空头,方向与 D1 对多头腿的结论相反。
| 目标(组合 OOS 2021–26) | ΔSR vs B [95%] | dev / test | 组合日内已实现波动 | MDD | 判定 |
|---|---|---|---|---|---|
| B 均值–方差 | —(1.766) | — | 21.9% | −16.9% | — |
| O2 下行半方差 | +0.068 [−0.048, +0.175] | +0.18 / +0.02 | 22.9% | −18.2% | 形式上满足(下界 −0.048 > −0.05),test 增量仅 +0.02,2025 −0.10 |
| O4 日内已实现方差 | −0.029 [−0.22, +0.16] | −0.04 / −0.02 | 21.8% | −17.4% | 不优于 |
O4 无效的原因(事后分解,`studies/ivol_posthoc.py`):组合日内已实现方差 76% 来自 QQQ 腿,Sleeve 空头腿与 QQQ 腿的交叉项为 −23%(日内对冲)。组合层的日内波动(21.9%)与日收益波动(22.1%)几乎相同,所以换成日内方差并没有改变风险项;Sleeve 的日内波动对收益有害(D1),但这是通过**期望收益**(来回止损)起作用,应该进特征/入场规则,而不是进风险项。

### B 部分(`studies/ivol_entry_study.py`)
自检:投毒 0.00e+00(首次运行发现 1e-16 差,原因是数组内存布局改变跨行求和的舍入顺序,已固定为 C 连续后复跑);阳性对照 IC 0.20(y)/ 0.24(y_ra)。y 与 y_ra 秩相关 0.93。
* 单变量:18 格中只有 `rv_surp` 对 y_ra 在两段都显著(IC −0.116 / −0.137),对 y 为 −0.08 / −0.12(test 显著)。其余 8 个特征的 |IC| ≤ 0.05。`rs_sgn`、`rskew_sgn`、`eff_k` 与 F1 的 `pos_dir`/`z_open` 等相关 0.5–0.9(重复信息)。
* 增量 ΔIC:ridge·y +0.072 [+0.020, +0.115];ridge·y_ra +0.035 [−0.003, +0.067];lgbm 两个目标区间都含 0。
* 经济(预注册 4 格,跳过 20%):Sleeve ΔSR 全 OOS −0.004 / −0.008 / +0.071 / +0.047,族内 p ≥ 0.12,组合 ΔSR 全部为负;**判定:无**。y_ra 两格的 test 期为正(+0.13 / +0.20)、dev 为负。

### 事后单格(`studies/ivol_posthoc.py`,累计 +1;由单变量表选出,18 选 1)
跳过 `rv_surp` 因果分位最高 20% 的笔:Sleeve ΔSR dev +0.48 / test +0.22 / 全 +0.31,单格置换 p = 0.001(按 18 格 Bonferroni ≈ 0.018);组合 ΔSR +0.09;被跳过的笔均值 −3.0bp(保留 +14.7bp);Sleeve 最差日 −228 → −174bp、CVaR5% −136 → −117bp。只跳空头时 dev/test 同正(+0.12 / +0.23),只跳多头 test 为负。
对策略的含义(假设,未检验):噪声带用过去 14 日同一时点的平均位移标定,不随当天波动调整;当天截至决策时点的波动明显高于同时点常态时,突破更可能只是噪声(D1 中给定位移日内波动有害与此一致)。下一步应预注册"按 rv_surp 放宽噪声带"的规则版本,而不是在过滤层继续挑。

## 偏离记录
* B 部分投毒自检首次运行出现 2.2e-16 差(`rv_surp`、`rv_com_dev`),定位为 F 连续数组下跨行求和顺序变化,改为 C 连续后为 0;特征定义未改。
* 事后追加 `ivol_posthoc.py`(分解 + rv_surp 单格),标注为事后,不改变上面的判定。配置累计:预注册 6 + 事后 1。

## 参考
* Andersen & Bollerslev (1997), *Intraday periodicity and volatility persistence in financial markets*, J. Empirical Finance —— [PDF](https://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/0/c9d1a098610563d0c12581f60072a23a/$FILE/AndersenBollerslev1997b.pdf)
* Patton & Sheppard (2015), *Good volatility, bad volatility*, REStat —— [PDF](https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf);Bollerslev (2022) 综述 [PDF](https://public.econ.duke.edu/~boller/Papers/Semi_JFEC_2022.pdf)
* Gao, Han, Li & Zhou (2018), *Market intraday momentum*, JFE —— [PDF](https://c.mql5.com/forextsd/forum/173/intraday_momentum_-_the_first_half-hour_return_predicts_the_last_half-hour_return.pdf);Baltussen et al. (2021) 归因于 gamma 对冲
* Barndorff-Nielsen, Kinnebrock & Shephard (2010) 已实现半方差;Amaya, Christoffersen, Jacobs & Vasquez (2015) 已实现偏度;Lo & MacKinlay (1988) 方差比
