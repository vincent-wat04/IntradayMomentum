# PROPOSAL — 多源新闻(multisource 语料)→ 次日日内波动 / Sleeve 好坏日(第四轮新闻研究,样本从 126 天扩到 ~1250 天)

状态:**预注册(执行前冻结)**。脚本 `studies/news_ms.py`(取数与特征)、`studies/lowvol_round4_news_study.py`(检验)、`studies/llm_news_ms.py`(第二阶段,需要 LLM 密钥)。
延续 `PROPOSAL_lowvol_avoidance.md`(第一~三轮)与 `PROPOSAL_llm_news_vol.md`:同一目标、同一 walk-forward、同一 A/B/E 模型,只把新闻特征 N 组换成多源语料。**不删改旧记录**。

## 0. 数据(HF bucket `vincequ/financial-news-multisource-bucket`,原 Ferrell 语料;仅限非商业研究)
本地缓存 `ETFT_DATA/raw/news_ms/`(1.3GB,不进 git)。使用 4 个子集的 **2019-01 ~ 2023-12**:
| 子集 | 性质 | 精度 |
|---|---|---|
| `sp500_daily_headlines` | 标普市场头条(每年 ~250 天) | 日级,带 `date_trading`(次一交易日) |
| `finsen_us_2007_2023` | 宏观/市场指标新闻(PMI、板块) | 日级,带 `date_trading` |
| `reddit_finance_sp500` | 散户讨论帖标题 | 分钟 |
| `yahoo_finance_felixdrinkall` | 财经/公司新闻标题 | 分钟 |
不用:NYT(非财经为主、2023 年中后断档)、`yahoo_finance_articles`(仅 2025)、`fnspid_news`(个股级)。2024 年起覆盖稀薄、2026 年无数据 ⇒ **新闻检验的 OOS 只有 2021–2023(≈750 天)**,训练从 2019 起。

## 1. PIT 规则(冻结,保守)
* 每条新闻归属『决策日 D』= 第一个满足 `D 的 09:30 ET ≥ 新闻时间` 的交易日;盘中(09:30–16:00)的新闻归次日。
* 日级子集:D = `date_trading` 当天(语料已按『下一个 NYSE 交易日』处理)。
* 新闻只用标题(文本第一行);`d` 日窗口 = 上一交易日 09:30 至 D 的 09:30。2023-12-29 之后一律置 NaN(覆盖不足,不能把『没有新闻』当成信息)。

## 2. 特征 N 组(冻结,14 个;全部非 LLM)
* 数量:`n_mk`(sp500+finsen 条数)、`n_rd`(reddit)、`n_all`(四源合计),均取 `log1p` 后减去**此前** 60 个交易日均值(shift 1)。
* 9 类关键词计数(`log1p` 后同上相对化):FED(fed/fomc/powell/rate hike/cut…)、INFL、JOBS、GEO(war/sanctions/tariff/…)、BANK(bank failure/default/…)、CRASH(plunge/selloff/panic…)、RALLY、EARN、TECH。正则与词表写在 `news_ms.py` 顶部,冻结。
* 情绪:`neg_mk`、`neg_rd` = 负面词/(正面词+负面词)(各 ~25 个词的固定词表),减去此前 60 日均值。

## 3. 检验(冻结,沿用 lowvol 第三轮)
* **目标/模型**:`lowvol_avoidance_study.build`+`walk_forward`(每年用上年末前数据重估);模型集 `A·ridge`、`A+B+E·ridge`(基线)、`A+B+E+N·ridge`、`A+N·ridge`、`N·ridge`。OOS = 2021–2023(dev 2021–22 / test 2023)。
* **主指标**:`A+B+E+N` 相对 `A+B+E` 的 ΔR²(log RV),月聚类自助 95% 区间;ΔAUC(低波标签)。
* **单变量**:14 个 N 特征与『A+B+E 残差』的 Spearman,dev/test 同号检查,**14 个检验按 Bonferroni 处理**。
* **变现**(仅当主指标过关才跑):按第三轮的 Sleeve 日级缩放(skip 20%/30%、连续 0.5/1.0),置换 p 的族内配置数累计到 50 + 4 × (模型数)。
* **有证据 ⇔** ΔR² 的 95% 区间不含 0 **且** dev 与 test 的 ΔR² 同为正 **且**(变现)组合 ΔSR 的族内 p<0.05。否则『没有足够证据』。
* 自检:污染测试在特征构造层——把 D 日 09:30 之后的新闻时间戳全部提前/改乱,D 日的特征必须不变;阳性对照:把『次日』新闻并入当日,特征必须变化。

## 4. 第二阶段(LLM 结构化抽取,`llm_news_ms.py`)
需要 SoCLaaS 密钥(**会话里没有保留,需用户重新提供;不写入文件**)。沿用 `llm_event_extract.py` 的冻结提示词 EXTRACT_V1 与聚合,输入改成 D 日窗口内 sp500+finsen 的头条(≤100 条/天,每批 20 条),模型 llama3.1:8b。特征 9 个(n_today_later/central_bank/macro_data/earnings/geopolitics/policy_trade/max_severity/n_sev2…),判定同 §3,累计配置数计入。**第二阶段只在第一阶段完成后、且用户提供密钥时运行;不依赖第一阶段结果,但与第一阶段分开报告。**

## 5. 先验(写在前面)
* 此前:GPR/EPU 文本指数 ΔR² +0.005 [−0.003,+0.013];LLM 整体评分/结构化抽取在 126 天上 OOS R² −0.37/−0.18(小样本)。
* 本轮样本 ~6× 大,检验功效够,但**关键词计数 ≈ 已有的事件日历 E 组(FOMC/CPI/NFP)的噪声版**;预计 ΔR² 区间含 0,单变量只有 FED/INFL/JOBS 可能同号弱相关;通过概率 ≤10%。
* 若有信号,最可能来自 `n_all`/`CRASH` 的『恐慌热度』,但这与 VIX(t−1)(A 模型已含)高度共线。

## 6. 偏离记录
**偏离 1(运行前,看数据覆盖后、未看任何预测结果)**:`yahoo_finance_felixdrinkall` 在 2021-01 从 ~2k 条/年跳到 ~20k 条/年(语料覆盖的结构性断点,且是公司级新闻),把它并入计数/类别会让特征随覆盖而不是随事件变化 ⇒ **剔除 yahoo**;关键词类别只在市场级来源(sp500+finsen)上计,reddit 另计占比(`rd_CRASH`、`rd_RALLY` = 占当日帖子比例)。N 组由 14 个变为 **15 个**:`n_mk, n_rd`、9 个 `mk_<类别>`、`rd_CRASH, rd_RALLY`、`neg_mk, neg_rd`。其余规则(PIT、相对化、OOS、判定)不变。

**偏离 2(实现细节)**:新闻只覆盖到 2023-12,故把 walk-forward 的重估年份限制为 2021–2023(`L.OOS_YEARS = range(2021, 2024)`);训练窗口与旧轮次相同(截至上年末),不影响 A/B/E 基线的定义。2024+ 的行因 W 缺失被排除。
**偏离 3(事后)**:第一阶段单变量里 GEO/BANK 过了 Bonferroni,我追加了一个『事后、探索性』检查(`news_ms_posthoc.py`,3 个附加配置,见下),这不改变预注册判定。

---
## 第一阶段结果(2026-10-12;`lowvol_round4_news_study.py`;日志 `studies/outputs/full/intraday/news_ms_r4_log.txt`;表 `outputs/tables/news_ms_{prediction,incremental,univariate}.csv`)
**自检**:把 2022-06-15 09:30 ET 之后的 54,324 条新闻时间戳整体后移 200~600 天 → ≤2022-06-15 的全部特征**逐位不变**(NaN 模式一致);阳性对照(盘中新闻并入当日)最大差 0.25>0 → PASS。
**样本**:新闻条目 118,617(使用 sp500 10,991 + finsen 10,032 + reddit 27,905 = 48,928 条;yahoo 69,689 条因覆盖断点未用)。OOS 753 天(dev 2021–22:503;test 2023:250)。

| 模型 | OOS R²(log RV)[95%] | AUC(低波) |
|---|---|---|
| A·ridge | 0.669 [0.583, 0.733] | 0.635 |
| A+B+E·ridge(基线) | **0.749** [0.674, 0.802] | 0.630 |
| W·ridge(新闻单独) | **−0.104** [−0.224, −0.0004] | 0.509 |
| A+W·ridge | 0.652 | 0.597 |
| A+B+E+W·ridge | 0.737 [0.663, 0.791] | 0.595 |
| A+B+E+W·lgbm | 0.691 | 0.605 |

**主指标**:`A+B+E+W − A+B+E` 的 ΔR² = **−0.0117 [−0.0217, −0.0018]**(dev −0.0142 [−0.0255, −0.0031];test +0.0016 [−0.0163, +0.0268]);ΔAUC −0.035 [−0.057, −0.014];lgbm 版 −0.058。逐年 ΔSSE:2021 −4.1%、2022 −10.2%、2023 +0.4%。
**即新闻特征不仅没有增量,反而使样本外变差(15 个特征 ridge 在 ~500 个训练日里学到了噪声)。**
**单变量**(特征 vs A+B+E 残差,Bonferroni p<0.0033):15 个中 2 个过线——`w_mk_GEO`(rho −0.138)、`w_mk_BANK`(−0.171),其余 |rho|≤0.074。**闸门未通过 ⇒ 按预注册不跑 Sleeve 变现。**

**事后检查(探索性,3 个附加配置;`news_ms_posthoc.py`,日志 `news_ms_posthoc_log.txt`)**:
* 在 A+B+E 上单独加 GEO:ΔR² −0.0004 [−0.0009, +0.0002];加 BANK:−0.0019 [−0.0060, +0.0013];两者联合:−0.0023 [−0.0066, +0.0011];dev/test 均不为正。
* 汇总的 rho 区间 [−0.239,−0.031]/[−0.263,−0.066] 不含 0,但**分年 rho 几乎为零**(GEO:2021 −0.07、2022 +0.01、2023 −0.08;BANK:2022 −0.01、2023 −0.02)⇒ 汇总相关来自**年份之间的水平差**(各年残差均值与特征均值不同),不是日度预测力。
* 新闻事件日确实偏热:BANK 最高的 15 天(集中在 2023-03 SVB)残差均值 +0.255 vs 全体 +0.100;GEO 最高的 15 天(2022-02 乌克兰)+0.119。但这些日子 A 模型(含 VIX(t−1)、前一日波动)已经基本吸收,且样本只有两三个事件簇。

**判定:没有足够证据。** 与 GPR/EPU(ΔR² +0.005 [−0.003,+0.013])、旧 LLM 整体评分一致:市场级新闻的**数量与关键词**在 VIX(t−1)+事件日历之上没有可用的日内波动增量。
**第二阶段(LLM 结构化抽取)未运行**:需要 SoCLaaS 密钥,会话里没有保留(见 §4)。提供后可在已缓存的 sp500+finsen 头条上直接跑(约 1~2 千次调用);按先验(关键词计数已无信息、旧结构化抽取 OOS 为负)通过概率 ≤10%,但样本已是原来的 6 倍。
**累计记账**:本轮配置 6 个模型集(A、A+B+E、W、A+W、A+B+E+W ridge/lgbm)+ 15 个单变量 + 3 个事后附加;未使用 Sleeve 变现的置换族。

---
## 第二阶段补充登记(2026-10-12,**运行前冻结**;用户重新提供了 SoCLaaS 密钥,仅经环境变量使用,不写入任何文件)
* 模型 `llama3.1:8b`(温度 0,单次);提示词 = `llm_event_extract.EXTRACT_V1`(**不改动**);输入 = D 日窗口内 sp500+finsen 的头条标题(按 §1 的 PIT 归属,经 `llm_news_vol_study.anonymize` 去日期/点位;≤100 条/天,每批 20 条)。2019-01 ~ 2023-12 的所有有新闻的交易日。
* 特征 9 个(`llm_event_extract.FEATS`:n_today_later、any_today_later、n_central_bank、n_macro_data、n_earnings、n_geopolitics、n_policy_trade、max_severity、n_sev2),与第一阶段一致地**相对化**(此前 60 日均值,shift 1;无新闻的天数视为 0 计数),2023-12-29 之后置 NaN。
* 检验、闸门、判定与 §3 相同(`lowvol_round4_news_study.evaluate`,标签 `_llm`);自检 = 第一阶段的 PIT 自检 + 解析覆盖率报告。
* 不换模型、不调提示词、不看结果后追加(旧记录:提示词/模型间不稳定;此处只跑一次)。累计配置:再加 6 个模型集 + 9 个单变量。
