> GitHub 版本说明：原研究报告保留于此供参考。仓库已收录全部核心代码与测试、六期限摘要结果；先前工作簿、原始研报附件和体积较大的逐日/全格点 JSON 未收录。可在提供行情后运行脚本重新生成。

# SPY / QQQ 日内量价因子与指数择时项目

本项目把日内量价研究转换成**单只 ETF 的时间序列预测**。输入是纽约本地时间、五分钟 bar 起始时标的 `datetime,open,high,low,close,volume`。公开样本覆盖 2024-06-03 至 2025-12-31，每标的 397 个交易日；2026 年尚未测试。下载的行情 CSV 不在交付包内。

**2026-10-04 六期限续篇：**逐期限筛选、滚动 ElasticNet/Ridge、每日重拟合 RAVEN-lite 的实际结果见 `results/multihorizon/MODEL_RESEARCH.md`，168 组逐日预测、IC/Rank IC/五桶/收益表见相邻 CSV 与 JSON。`raven_paper_model.py` 实现论文结构，但本次环境没有可安装的 PyTorch，完整网络**尚未训练、尚无实测预测结果**；状态单独记录在 `raven_full_status.json`。因此不应把续篇称为完整论文收益复现。

## 项目文件

| 文件 | 用途 |
|---|---|
| `index_timing_framework.py` | 原有噪声带因子、日频聚合、滚动正交、事件与组合回测、分桶单调性 |
| `extended_timing_research.py` | 新增 27 个量价代理和原有 2 个基线，全部窗口/期限、两种执行的因子实验 |
| `raven_lite_timing.py` | 小样本、自适应窗口的 **RAVEN-inspired 诊断模型**，时间切分与样本外交易模拟 |
| `multifrequency_models.py` | 逐期限冻结筛因子、多 K 选择，滚动 60/120 日及扩张 Ridge/ElasticNet、更新版 lite |
| `raven_paper_model.py` | PyTorch 论文结构实现（16 日 patch、三专家、CIT/CAW/GCR；本次未能运行训练） |
| `write_model_report.py`, `test_multifrequency_models.py` | 续篇报表生成、标签成熟及防前视检验 |
| `test_index_timing_framework.py`, `test_extended_research.py` | 时间对齐、历史分桶、模型路由与执行逻辑校验 |
| `build_close_execution_workbook.mjs` | 从**原有框架的 JSON** 导出分层 XLSX；不是新因子 JSON 的生成器 |
| `README_index_timing.md` | 原有框架的因子公式、回测口径和已知数据问题 |
| `FACTOR_CATALOG.md` | 新因子逐项动机、公式和源文献映射 |
| `results/` | 新因子的完整 JSON、模型输出与前期分层工作簿 |
| `legacy/` | 历次探索脚本；其中 `index_timing_probe.py` 已被替代，不作为主结果 |

## 安装和运行

Python 3.10+；`numpy pandas scipy scikit-learn`。模型脚本只用 NumPy 与 scikit-learn；**没有 PyTorch 论文复现**。例如：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest test_index_timing_framework.py test_extended_research.py
python index_timing_framework.py --spy /path/SPY_5min.csv --qqq /path/QQQ_5min.csv --output base_results.json
python extended_timing_research.py --spy /path/SPY_5min.csv --qqq /path/QQQ_5min.csv --output-dir new_results
python raven_lite_timing.py --spy /path/SPY_5min.csv --qqq /path/QQQ_5min.csv --output new_results/raven_lite_results.json
python multifrequency_models.py --spy /path/SPY_5min.csv --qqq /path/QQQ_5min.csv --output new_results/multihorizon.json
# PyTorch 是完整网络的额外依赖；安装后：
python raven_paper_model.py --spy /path/SPY_5min.csv --qqq /path/QQQ_5min.csv --output new_results/raven_full.json
```

所有路径都可以在命令行指定。原工作簿生成器需 Codex artifact-tool 的 Node 环境，调用：`node build_close_execution_workbook.mjs base_results.json base_results.xlsx`。普通本地 Python 回测不依赖它。

## 研究口径

设日内已形成因子为 `f_t`，窗口 K 取 `1/5/10/20/30/60`：普通因子 `F[t,K]=mean(f[t-K+1:t])`；名称 `_VAR` 的“稳定性”因子为同一窗口的总体方差 `mean((f-mean(f))²)`，K=1 无定义。**每个 K 单独正交**：控制 `C_t/C_{t-K}-1`，用截至 t−1 以前最多 120 对样本、至少 60 对含截距 OLS 得到 t 日残差。`raw_matched` 仅保留残差可得日期以作配对比较。截距、斜率和历史分位桶均不读取未来日期。

两种执行：T+0 表示 t 信号后在 t+1 开盘假设成交，未来 h 日收益 `C[t+h]/O[t+1]-1`；T+1 表示 t+1 收盘成交，收益 `C[t+1+h]/C[t+1]-1`。期限 `1/5/10/20/30/60/90/120`。输出同时含按各期限可评估日期和所有期限共同 120 日截止的两套结果。每行有 Pearson IC、Spearman、按 h 阶 Newey–West 的斜率 t 值、原始收益、上涨比例、五桶均值/计数/贪婪非重叠次数、高减低及相邻桶上升数，并按信号年份列相关与桶计数。历史分桶只用 t−1 以前最多 120 个有效因子值（至少 80 个），每桶观测数可能不均。

五桶的高减低是**不同日期的条件均值差**，不能解释成同一天可交易的多空收益。重叠的 60 日目标高度相关；高绝对值 IC 可由趋势状态和可评估年份集中造成。只读五桶均值不足以确认可投资性。

## 模型诊断与发现

`raven_lite_timing.py` 不是 RAVEN 原模型：它保留实例时间标准化、五日 patch、CIT 三个嵌套窗口、各窗口专家、余弦冗余权重和全窗口支路的**结构思想**，使用正则化线性专家，不含原文可训练 Transformer、Self-Attention、联合反传、熵/多样性正则。因此结果标作 `RAVEN-lite`，不能宣称论文复现。选择 10 个预先指定的 5 日正交因子；历史上下文 20 日。2025-03-03 前的成熟标签作训练，至 2025-06-02 前已成熟的标签作验证，2025-06-02 以后才为锁定测试。验证仅选择 3 个打分权重 × 3 个 ridge 惩罚；随后使用测试开始之前成熟的训练+验证标签重拟合。各执行版本分别用其真实入场、退出目标训练，T+0 标签在 t+1 收盘成熟，T+1 在 t+2 收盘成熟。样本外比较固定窗口 ridge 和在验证期选出的最佳单因子。历史缺失值仅向前最多填 5 日，预测日不能是早收盘日。

这轮模型的测试期均有 143 个信号日，初始训练只有 78–79 日。**SPY/QQQ 两种执行的测试期预测 Pearson 都为负**（见 JSON），所以没有验证出复合 alpha；部分 long/cash 策略虽优于两个简化比较模型，仍低于同区间买入持有，不能解读为更强的可交易预测。若要正式训练和复现 RAVEN，需更长的复权且可交易的 SPY/QQQ 历史、真实收盘/开盘成交数据、多市场状态的独立锁定测试集，以及 PyTorch 和论文级网络训练；不得靠重复查看同一测试段选参。

## 数据与来源边界

- 原始数据仓库：<https://github.com/Aryan092/fds-indx-data>。该数据末根五分钟 `close` 未核验为收盘撮合价，未作 ETF 分红复权；早收盘按配置表处理，因子只在完整交易日形成。统计上的 t+1 价格映射不等于已验证的实际成交。
- 用户提供的东北证券《日内成交量分布因子及 Logsig-Alpha 因子生成》（2023-11-29），其中 16 个手工因子是横截面选股；本项目测试同名概念的 ETF 时间序列变体。价格区间因子用 bar **开盘价**承载该 bar 成交量，属于代理而非逐笔价量分布。报告的 20 日 lead-lag 十阶 log-signature + 截面 MLP 因子**未声称复现**：缺少成千股票的横截面训练面板和原架构实现。
- [广发金工文章](https://www.sohu.com/a/951814256_121118711) 研究的是对**已有股票多空因子收益**的 92 种择时信号；只有动量、波动率和反转概念能在单 ETF 价格序列上形成可观察的代理，Barra 多空组合特征差、Level-2/深度学习股票因子收益不可从两个 ETF OHLCV 算出。
- [Green, Hand & Zhang (2017), RFS](https://doi.org/10.1093/rfs/hhx019) 研究 94 个股票横截面特征；只借鉴价格反转、最大收益、波动与成交金额/非流动性概念。未计算财务报表、市值、股份换手、机构持有等缺失字段，也不把两个 ETF 假装成全市场截面。
- [RAVEN 原论文](https://arxiv.org/abs/2606.24062) 使用 2009–2024 数千股票的日度训练样本、可训练专家与独立市场年份测试；本项目自适应窗口实现及结果不能直接比较论文中的横截面 ICIR。

这是一组探索性代理实验，29 个因子 × 6 个窗口 × 多期限的尝试具有显著多重比较风险。下一步应冻结少量候选，用更早起点和 2026 完全未触及的独立样本重新测试，并取得股本、GEX、期权、财务与风格数据后再扩展对应因素。
