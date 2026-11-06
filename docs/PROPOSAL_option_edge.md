# 预注册草案:用波动率预测在 0DTE 期权上找 edge(文献综述 + 设计,未运行任何回测)

2026-10-19。状态:**草案,待用户确认后冻结**。冻结之前不看任何期权数据的收益。
数据:`optrec` 自 2026-10-19 起录制 SPY/QQQ 0DTE+1DTE 链(10 秒快照,只有买卖价/IV/Greeks,无逐笔)。

## 1. 文献读到了什么(以及对我们的含义)

| 文献 | 结论(转述) | 对本研究的含义 |
|---|---|---|
| Vilkov (2026),*0DTE Trading Rules*,SPXW,2016-09~2026-01,10:00 入场持有到收盘 | 0DTE 方差风险溢价为正但同日量级很小;各结构收益分布宽、厚尾;**盈亏更多由已实现偏度(方向)驱动,而不是已实现方差**;条件择时把问题写成"方向分类"比直接预测收益有效;计入半价差+0.5bp 后:strangle/straddle 条件 OOS Sharpe 毛 0.56 / 净 0.39,铁蝶/铁鹰毛 0.77 / 净 −0.20,put ratio spread 毛 1.18 / 净 0.93 | ①"预测波动 > IV 就买波动"本质是方差押注,文献里方差只能解释很小一部分 PnL;②成本能把毛 edge 吃掉一半以上;③方向信号比波动信号更有用 → 我们应该把**方向**(首 30 分钟突破)作为主信号,把波动预测当作"是否值得付权利金"的过滤器 |
| Coval & Shumway (2001) | 零 beta 的平值跨式买方平均每周亏约 3%(买波动的无条件期望为负) | 无条件买入必亏,必须有**条件性**的 edge 才可能转正 |
| Goyal & Saretto (2009) | 按"历史已实现波动 − 平值 IV"排序,买高差/卖低差的跨式组合月度收益显著 | 我们的信号形式(预测波动 − IV)有先例,但它是月度、横截面、股票期权;**日内单标的不能直接外推** |
| Kanniainen & Magris (2018) | 指数期权的 IV 在收益跳跃后会**延迟、渐进**调整,下跌跳跃后更明显;但计入真实买卖价差后没有异常收益 | 支持"突破后再买"有逻辑,但同时警告:价差会吃掉它 |
| Muravyev & Pearson (2020);Barardehi-Bogousslavsky-Muravyev | 有效价差远小于报价价差(择时成交可降到不足 40%;散户实际成本比常规有效价差低约 60%) | 成本不能只按"全价差"悲观估计,也不能默认中间价成交。回测必须做**价差分数敏感性**(100% / 60% / 40%),并以 100% 为主结论 |
| Londono & Samadi / *Event-day options* (NBER w28306) | 事件日(尤其 FOMC)方差风险溢价大且显著为正,即事件日期权被高估 | 你提到的宏观事件调整:**事件日买波动逆风最大**(IV 已含事件溢价,事件后 IV 压缩)。事件日默认不交易或单独成一个预注册分支 |
| Gao-Han-Li-Zhou (2018);Baltussen et al. (2021);Da et al.(期权跨式日内动量) | 首半小时预测末半小时;来源于期权/杠杆 ETF 的 gamma 对冲;跨式收益在同一半小时区间存在跨日动量,开盘和收盘尤其明显 | 方向信号(首 30 分钟)有理论依据;但那是**现货**的末半小时,不是整个下午。期权要赚钱需要走得比 IV 隐含的还多 |
| Dim-Eraker-Vilkov (2024) | 0DTE 未平仓 gamma 不会向外传导放大波动 | **不能**用"0DTE 对冲流放大波动"当作买波动的理由 |
| Bandi-Fusari-Gazzani-Renò (2026) | 超短期平值 IV 期限结构震荡明显,经典模型难以同时拟合 | 不要依赖供应商给的 IV/Greeks(盘后已见到明显异常),**用买卖价中间价自己反推** |

## 2. 与我们已有结论的冲突:必须先面对

1. **位移预测 R² 仅 0.05~0.09,R² 0.749 的振幅预测变现不了**(`PROPOSAL_vol_monetization.md`)。高 R² 来自持续性,而 IV 同样包含持续性。所以"预测波动 > IV"如果只是"今天波动大",市场价格里已有。需要的是**对 IV 的增量信息**。
2. 文献普遍显示"买方 + 毛 edge 薄 + 价差大"。我们的先验应是**没有 edge**,研究的价值在于干净地证伪或证实。
3. **样本量**:每个标的每天最多 1 笔。设单笔收益 SD 约等于权利金(100%,假设),要在 80% 功效、双侧 5% 下检出 +10% 的平均收益需约 780 笔,检出 +20% 需约 200 笔。录制 3 个月只有几十笔,**不够下结论**。

## 3. 数据路线

- A. 自录(optrec):免费,但样本是"从现在起"。适合验证**微观结构**(价差、IV 质量、成交可行性),不适合验证 edge。
- B. 购买历史 NBBO:Databento 的 OPRA 数据集提供 1 分钟 CBBO(2013 起)和 1 秒 CBBO(2023-03-28 起),覆盖 SPY/QQQ 期权;ThetaData 也提供带 IV 的逐笔/报价历史。**价格我没有核实**,需要你确认预算。有了历史 NBBO,就可以用 2023 年以来的数万笔 SPY/QQQ 日内期权做预注册检验,样本问题基本解决。
- C. 券商历史 K 线(moomoo):"已到期合约能否拉到历史 1 分钟 K 线"还没验证。这会占用与 longvol/live_noise 共用的 300 标的历史 K 线额度,**未经你同意我不会去试**。

建议顺序:先 B(哪怕只买 SPY+QQQ 一年)做离线检验 → 通过后再用 A 做前向影子验证。

## 4. 期权经济学:怎么判断"便宜"

不依赖供应商 IV。三个量,全部由买卖价算出:

1. **市场隐含的剩余日内预期绝对波动**:`M_mkt = 平值跨式中间价 / (0.798 · S)`(平值跨式 ≈ 0.798·σ·√T·S 的近似,忽略利率与偏斜;由两个最近平值行权价线性内插到 S)。
2. **预测的剩余日内预期绝对波动**:`M_fcst`,由现有波动预测模型给出,换算到 10:00~16:00 视野。**换算方法必须在冻结前定死**,并且要用与 `M_mkt` 同口径的"|S_T − S_t|"目标。
3. **单个期权的预测 EV 比**:`edge = E_fcst[payoff] / ask − 1`。其中 payoff 分布由"预测标准差 × 经验标准化残差"(用 QQQ 1 分钟历史标定,保留厚尾)生成,不使用高斯。行权价选择:在满足最小权利金、最大相对价差(如 ≤15%)的候选里取 `edge` 最大者。

**主检验(阶段一,零交易)**:`(RV_剩余 − M_mkt) ~ (M_fcst − M_mkt)`,日期聚类标准误。斜率显著为正且 > 0,才有资格进入阶段二。这一步直接回答"预测模型相对期权市场有没有增量信息",也是文献里 Goyal-Saretto 的日内版本。

## 5. 交易结构(阶段二,仅当阶段一通过)

只预注册 3 个结构,避免分叉路径:

- **S1 单边**:10:00 按首 30 分钟方向买入一档虚值期权(目标 |delta| 约 0.25~0.35)。
- **S2 单边 + 反向小翼**:S1 加买反向、更远虚值的小仓位(权利金约为 S1 的 1/4),用于限制"突破失败反向走"的亏损。依据:Vilkov 中比例价差/风险逆转类结构在方向条件下更稳,但那是卖出结构,**买入翼是否划算本身是待检验的问题**,不预设有效。
- **S3 对照**:平值跨式(纯波动押注),用来检验"方向"是否真的比"波动"重要。

规则(先验给定,不在小样本上调参):
- 入场 10:00:仅当 `M_fcst − M_mkt` 的 walk-forward 分位数高于阈值,且首 30 分钟 |return| 大于其滚动中位数。
- **10:30 检查**:若 |S − S_10:00| 小于 `0.3 · M_mkt · S`(价格没有走出来),双腿全平。
- 止盈:方向腿权利金 +100% 平一半,其余追踪;时间止损:15:30 前无条件平仓(避免最后半小时 gamma/价差与 pin 风险)。无价格止损(已是有限风险买方)。
- **事件日**(FOMC/CPI/NFP):默认不交易;若要交易,10:30 检查改为事件发布后 15 分钟,并作为独立分支单独报告。
- 成本:报价价差 100% 半价差为主结论,60%/40% 为敏感性;外加每张手续费。

## 6. 成功标准(提前写死)

- 阶段一:斜率 > 0 且日期聚类 t > 2,且 dev/test 同号。
- 阶段二:净于 100% 半价差后,S1 或 S2 的平均每笔收益 > 0,置信区间下界 > 0(笔数 ≥ 200),并报告完整分布(偏度、CVaR5%、最差笔、前 5% 笔贡献利润占比)。
- 家族:3 结构 × 2 标的,按 6 个检验校正。任何事后的阈值/行权价调整都标为事后。
- 否定也是结果:阶段一不通过则停止,不进入交易模拟。

## 7. 个股(你问到的)

大市值个股一般只有周度到期(周五才是 0DTE),不是每日末日期权。纳入它们需要重做合约池和快照预算。**建议 SPY/QQQ 先跑通,个股作为第二阶段**,并且只选期权价差窄、成交活跃的少数几只。

## 8. 待你决定

1. 是否购买历史 NBBO(Databento/ThetaData),预算多少?这决定研究周期是几周还是一年以上。
2. 是否允许用一个历史 K 线额度试 moomoo 已到期期权合约(C 路线)?
3. 同意冻结本草案后,我会先做阶段一的数据管线和与现有波动预测模型的对接。

## 参考
- Vilkov (2026), *0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing*, SSRN 4641356;代码 https://github.com/vilkovgr/0dte-strategies
- Coval & Shumway (2001), *Expected Option Returns*, JF
- Goyal & Saretto (2009), *Cross-section of option returns and volatility*, JFE
- Kanniainen & Magris, *Option market (in)efficiency and implied volatility dynamics after return jumps*, arXiv:1810.12200
- Muravyev & Pearson, *Options Trading Costs Are Lower than You Think*, RFS
- Bogousslavsky & Muravyev, *An Anatomy of Retail Option Trading*
- *Event-day Options*, NBER w28306;Londono & Samadi, *The Price of Macroeconomic Uncertainty: Evidence from Daily Options*, Federal Reserve IFDP 1376
- Gao, Han, Li, Zhou (2018), *Market Intraday Momentum*, JFE;Baltussen, Da, Lammers, Martens (2021), *Hedging Demand and Market Intraday Momentum*, JFE;Da et al., *Momentum in the Option Market, Intraday Case*
- Dim, Eraker, Vilkov, *0DTEs: Trading, Gamma Risk and Volatility Propagation*
- Bandi, Fusari, Gazzani, Renò (2026), *Ultra-short-term volatility surfaces*, arXiv:2603.29430
- 数据:Databento OPRA(CBBO-1m 自 2013、CBBO-1s 自 2023-03-28)https://databento.com/blog/opra-improvements-coming-soon

内容为转述,未逐字引用。Vilkov 的数字来自其公开 GitHub 版本的论文文本,未独立复现。
