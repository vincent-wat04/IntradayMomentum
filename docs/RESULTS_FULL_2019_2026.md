# 完整数据复跑：因子回测 + 模型复合因子（SPY / QQQ，2019-01 → 2026-10）

## 1. 数据与改动
- 数据：moomoo 1 分钟线转为 5 分钟线（起始时标、纽约时间），1948 个交易日，含 15 个半日市，无缺失交易日。用 Yahoo 复权因子做**后复权**（原公开数据未复权）。脚本：`scripts/build_5min_from_moomoo.py`。
- 代码改动最小：因子定义、正交化、IC/HAC/分桶、`screen_factors`、Ridge/EN/RAVEN-lite/RAVEN 均沿用仓库代码。只改了：
  1. 早收盘日历由 `data/early_closes.txt` 提供；
  2. 测试起点 `TEST_START = 2023-01-03`。筛选、宏观预筛只用 2019–2022 年已成熟标签；
  3. 新增宏观因子模块 `macro_factor_study.py`，以及 B 特征集。
- 新增脚本：`run_full_extended.py`、`macro_factor_study.py`、`run_full_models.py`、`analyze_full.py`。结果在 `outputs/full/`，汇总在 `outputs/full/report/`。

## 2. 单因子回测
- 日内/量价：29 个因子 × 6 个 K × 3 个版本 × 2 种执行 × 8 个期限，共 32,256 行。
- 宏观：从 etf_timing 导入 102 个市场层面因子（波动率、相关性、信用、利率、宏观、情绪、NLP、仓位、事件等，均 PIT），共 52,224 行。`hy_oas_chg_21` 和 `put_call_total_10` 覆盖率不足，已剔除。
- "稳定"的定义：SPY、QQQ 两个标的在 2019–22 和 2023–26 两段的 IC 都同号。

| 期限 | \|HAC t\|>2 占比 | 稳定且 \|t\|>2 的个数 |
|---|---|---|
| 1 | 4.0% | 43 |
| 5 | 12.2% | 122 |
| 20 | 20.8% | 191 |
| 30 | 23.4% | 214 |
| 60 | 21.3% | 183 |

h=1 时占比接近随机水平（约 5%），说明日频几乎没有信号；随期限变长显著比例上升，但长期限样本重叠严重，t 值有高估。

**最强、最稳定的一类是"恐慌补偿"类宏观因子**（`factor_top.csv`）：
- VIX 期限结构 `vix_ts_6m`/`vix_ts_3m`：h=60 IC 约 0.36–0.40，t ≈ 5；
- 隐含相关性 `implied_corr_3m`、`vixeq_ratio`：h=20–30 IC 约 0.25–0.30，t ≈ 4.2–4.7；
- 其他：OVX、`baa_spread`、VVIX、`absorption_ratio`。

方向都是"恐慌越高，后续收益越高"，而且 2023–26 年比 2019–22 年更强。这与 etf_timing 在 2000–2021 年的独立结论一致。

日内量价因子里，`OPEN_VOL_SD(_VAR)`、`PM_MID_CLOSE_RATIO(_VAR)`、`VP_REVERSAL`、`LOGVOL_STD` 是负向且稳定的，t 约 2.5–3.4。按族统计，稳定显著的条目里 volatility、correlation、intraday report 三族最多（`factor_stable_by_family.csv`）。

要注意的是：`cape_ey` 和 `profits_yoy` 的 t 也很高，但它们只在 2019–22 年有效，2023–26 年 IC 接近 0。

## 3. 好因子之间的相关性
图：`good_factor_corr_{SPY,QQQ}.png`。取 24 个稳定且 |t|>2 的因子（用正交版本，即模型实际输入的形式），每个因子保留最好的 K。

- **平均 |相关| 为 0.25；有效独立因子数只有 6.8（SPY）/ 7.1（QQQ）。**
- 高相关簇：
  - `implied_corr_1m` 与 `vixeq_ratio` 相关 0.93；
  - `iv_level`、`implied_corr_3m`、`baa_spread`、`OVX`、`VVIX` 两两之间 0.70–0.86，本质上是**同一个"恐慌/风险溢价"因子**；
  - `breadth_hilo` 与 `breadth_above_50` 相关 0.72。
- 日内量价因子和宏观簇之间基本正交，这是它们真正的增量来源。

模型实际选中的特征（并集）：`selected_factor_corr_*`。平均 |相关| 为 0.15–0.18，高相关主要是**同一因子取不同 K**（例如 `esi_growth:1` 与 `:5` 相关 0.95）。由于每个期限、每种执行方式是分开筛选的，单个模型内部已经剔除了相关性大于 0.85 的特征。

## 4. 模型复合因子（测试期 2023-01 → 2026-10，约 870–930 个信号日）
图：`model_ic_heatmap.png`、`equity_h{1,5,20}.png`；表：`model_summary.csv`、`model_overview.csv`。

| 模型 | 平均 IC（A / B） | 2023–24 / 2025–26（B） | 正向显著个数（B） | 多头/空仓 Sharpe（B） | 买入持有 Sharpe |
|---|---|---|---|---|---|
| Ridge 扩张窗口 | +0.083 / **+0.142** | 0.092 / 0.202 | 4 | 1.39 | 1.47 |
| EN 扩张窗口 | +0.080 / **+0.137** | 0.087 / 0.196 | 4 | 1.41 | 1.47 |
| Ridge/EN 60 日窗口 | −0.11 / −0.10 | | 0 | ~0.97 | 1.47 |
| Ridge/EN 120 日窗口 | −0.06 / −0.08 | | 0 | ~1.15 | 1.47 |
| RAVEN-lite | −0.044 / −0.009 | | 1 | 1.27 | 1.47 |
| RAVEN（论文结构） | +0.012 / −0.035 | | 0 | 1.19–1.33 | 1.47 |

- **扩张窗口明显好于滚动窗口**。60/120 日窗口的 IC 系统性为负，40 个显著条目中多数是负的（在追逐上一段行情）。在这种低信噪比下，估计误差比非平稳性更致命。
- **深度模型没有带来增益**。RAVEN 每 63 天重训一次，训练样本约 1000 个，平均 IC 约 0。只有 h=1 QQQ（−0.08，t −2.4）和 h=20 QQQ（−0.11）显著，而且都是负的。
- **加入宏观特征（B 相对 A）**：总体 IC 差 +0.008，55% 的组合有改善，Sharpe 差 −0.01，整体没有增量。但在扩张窗口线性模型上 IC 提高了 +0.06，2025–26 年 IC 达到约 0.20，属于局部有效。
- **最强的结果**：QQQ h=60，扩张窗口 Ridge/EN，B 特征集，IC 0.55，HAC t ≈ 9–10，两段时期分别为 0.34 和 0.68。SPY h=30 扩张窗口 IC 约 0.30–0.34，t ≈ 2.5。
  对此要谨慎：60 日标签在 3.8 年里只有大约 15 个不重叠的样本；而且模型 88% 的时间都是多头，多头/空仓 Sharpe 1.73，买入持有 1.56，只高一点。这更像是"牛市里大多数时候持有、少数时候离场"的效果，而不是独立的择时能力。
- **可交易性**：所有模型在测试期的多头/空仓 Sharpe 平均都**低于买入持有**（1.47）。最好的是扩张窗口 EN 的 A 集，58% 的组合跑赢买入持有。

## 5. 结论
1. 完整数据下，单因子层面**最稳定的信号来自宏观的恐慌补偿簇**（期限结构、隐含相关性、信用利差、OVX）。但这一簇的有效独立维度大约只有 1 个。整个好因子池的有效独立因子数约为 7。
2. 宏观因子"择优加入"模型：在**扩张窗口线性模型**上 IC 从 0.08 提高到 0.14；对 RAVEN 系列和滚动窗口模型没有帮助。建议保留 B 集，只配合扩张窗口的 Ridge/EN 使用。
3. 复合模型在 2023–26 年（强牛市）的多头/空仓策略**没有跑赢买入持有**；高 IC 主要集中在 h≥30 的长期限，而这些期限的独立样本很少。
4. 多重检验：8 个模型 × 6 个期限 × 2 种执行 × 2 个标的 × 2 个特征集 = 384 组。单个组合 |t|>2 的证据力度要大打折扣。

## 6. 复现
```bash
python scripts/build_5min_from_moomoo.py --etf-timing ../etf_timing --out data
python index_timing_framework.py --spy data/SPY_5min.csv --qqq data/QQQ_5min.csv --output outputs/full/base_results.json $(sed 's/^/--early-close /' data/early_closes.txt)
python run_full_extended.py && python macro_factor_study.py
python run_full_models.py --stage linear --horizons 1 5 10 20 30 60
python run_full_models.py --stage raven --horizons 1 --raven-epochs 20 --raven-retrain 63   # 每个期限分别跑，约 13 分钟/期限
python analyze_full.py
```

## 单因子 Sharpe（`factor_sharpe.py` + `factor_sharpe_report.py`）

**方法（无未来信息）：** 信号是因子在过去 252 日窗口内的分位数（只用当时可得的数据），方向由 2019–22 的 IC 符号固定，再做 h 日平滑。next_open 执行，单边成本 1bp。共三种仓位：
- **LS**：仓位 ∈[-1,1]，纯择时多空；
- **LC**：仓位 ∈[0,1]，多头或空仓；
- **TILT**：仓位 0.5+0.5·信号。

每个 (标的, h) 有 576 个因子列，h 取 5 和 20。dev 为 2019–22，test 为 2023-01-03 之后。

### 分布（测试期）
| | LS 中位 SR | LS p90 | LS SR>0 占比 | LC 中位 | TILT 中位 | TILT 跑赢 B&H 占比 | B&H SR |
|---|---|---|---|---|---|---|---|
| SPY h5 | 0.09 | 0.61 | 60% | 1.00 | 1.23 | 21% | 1.40 |
| SPY h20 | 0.13 | 0.64 | 64% | 1.04 | 1.25 | 25% | 1.40 |
| QQQ h5 | 0.04 | 0.60 | 55% | 1.07 | 1.30 | 18% | 1.50 |
| QQQ h20 | 0.09 | 0.66 | 58% | 1.08 | 1.33 | 23% | 1.50 |

- dev SR 与 test SR 的截面相关只有 0.02–0.16（见 `factor_sharpe_dev_vs_test.png`）。也就是说，单个因子的历史 Sharpe 几乎不能预测它未来的 Sharpe。
- LC/TILT 的 Sharpe 主要来自"平均持有约 0.5 倍多头"的贝塔。在 2023–26 牛市里，它们很少能跑赢 B&H。

### dev 筛选（两只标的的 dev SR 都 >0.5）
- **LS：** h5 选出 66 个，h20 选出 65 个。测试期中位 SR 为 0.09–0.26，65–80% 为正，比全体因子略好。
- **等权组合 LS（测试期）：**

  | | SR | 年化 | 波动 | MDD | 与 B&H 相关 |
  |---|---|---|---|---|---|
  | SPY h20 | 0.97 | 2.2% | 2.3% | -2.8% | 0.27 |
  | QQQ h20 | 0.86 | 2.5% | 2.9% | -4.4% | 0.20 |

  分散化把单因子约 0.2 的 SR 提升到了约 0.9，但收益和波动都很小（因子之间的多空信号大量相互抵消）。
- **叠加：** B&H + 0.5×（按 dev 期波动匹配的 LS 组合，k≈4–5，总杠杆最高约 3x）。
  - SR：SPY 1.40→1.52，QQQ 1.50→1.60；
  - 年化：SPY +5.7pp，QQQ +7.4pp；
  - MDD：SPY 恶化约 1.7pp，QQQ 恶化约 2.2pp。
  - 见 `factor_sharpe_overlay.png`。
- **稳健的 LS 单因子（测试期 SR，SPY/QQQ）：**
  - VP_REVERSAL:1 h20：1.09 / 1.02（波动仅约 2%）
  - [M]vxn_minus_vix:5 h20：0.77 / 1.01
  - [M]tom_next_week:20 h20：1.02 / 0.51
  - [M]cape_ey:1 h5：0.57 / 0.55
- **测试期反转的因子：**
  - [M]days_since_fomc：dev 1.2 → test −0.4
  - [M]fomc_cycle_even 也反转了
  - [M]esi_growth 降到约 0

**注意：** 共 576 列 × 2 个 h × 3 种仓位 × 2 只标的 ≈ 6900 次检验。在多重检验下，单个因子的 test SR≈1，其 t 值约为 1.9（3.75 年），不显著。只有等权组合的结论比较稳健。

图：`factor_sharpe_dev_vs_test.png`、`factor_sharpe_tearsheet_{SPY,QQQ}.png`、`factor_sharpe_overlay.png`。表：`factor_sharpe_summary.csv`、`factor_sharpe_composites.csv`、`factor_sharpe_dev_selected.csv`。

> 后续：原始因子的时间聚合、标准化和合成方法的系统比较（包括零假设对照，以及噪声带中位数/偏度因子的结论），见 `docs/TIMING_TRANSFORMS.md`。
