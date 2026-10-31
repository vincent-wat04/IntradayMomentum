# Noise Band 设计审查与实验实现记录

日期：2026-10-31
分支：`research/noise-band-validation-2026-10`

## 审查结论

### 原始设计

`studies/etft/intraday/noise.py` 正确实现了 Zarattini、Aziz、Barbon 的基本定义，并通过 `shift(1)` 只使用过去交易日。`studies/etft/intraday/strategy.py` 在决策 bar 收盘观察信号、下一 bar 开盘成交。现有 audit 的未来投毒和独立账本对账没有发现机械性前视。

但 `mean(|close/open-1|)` 是条件绝对位移的点估计，不是预测区间。仓库此前没有输出 coverage、coverage drift 或 conditional coverage，也没有检验“带外”相对同等幅度的带内事件是否有额外延续性。因此代码中的 “noise area” 是论文术语，不是已被当前数据统计确认的对象。

### vol / turnover 改进

最新日内设计不是 `noise.py` 的 `rvol^0.5` 或 volume clock，而是 v4/v5 使用的 entry-only `s3`：累计 realized variance surprise 除以累计 volume surprise，再开平方并截在 `[1,2]`。止损仍使用原 band。这比第二轮同时放宽入场与止损更合理，因为它避免主动放松风险退出。

不过现有历史结果不支持“已证实改善”：

| 证据 | 结果 | 解读 |
|---|---:|---|
| C3，vol/volume 同时调整入场和止损 | ΔSR +0.039，95% `[-0.138,+0.201]` | 区间跨 0 |
| E1，只调整入场 | ΔSR +0.039，95% `[-0.138,+0.201]` | 没解决统计不确定性 |
| E1 Sleeve 最差日 | 约 -332bp，原基准约 -250bp | 左尾恶化 |
| E1 后的 `rv_surp` 特征替换 | 单步 ΔSR -0.073 | vol/volume 信息不是简单可加 |
| v5 test Sharpe | 2.87 | 包含核心仓、风险配比和买力约束，不能归因给 band |

总成交量是 activity proxy，不是订单流方向。仓库目前没有 bid/ask、主动买卖方向、盘口深度或逐笔冲击，因此“turnover 支持的信息冲击”只能作为假设。

本次还发现两个会影响“当前 v5”比较口径的问题：

1. `tearsheet_v5.build_v5` 用当日 10:00 后才得到的 `kmax` 计算当日 cap 缩放，再把缩放权重乘到整日核心仓收益，因而原 v5 组合存在候选相关的日内前视；
2. v4/v5 用 `G_series` 从 MOC P&L 中减去 15:40 后价格段来模拟提前平仓，但没有把 MOC 退出成本替换成 15:41 主动退出的 half-spread，成交明细仍显示 MOC。

新实验不改历史产物，而是在比较层使用开盘前已知的 `k=2` 上界计算 cap，并补上 15:41 主动退出成本。因此新结果会标作 **PIT-corrected v5 comparison**，不要求逐值复现 `tearsheet_v5_steps.csv`。

## 全部分支与 PR 结论

检查到的远程分支：

- `main@574cbcbe`
- `research/layered-2026-10@3f5902b4`
- `research/duration-vol-entry-2026-10@bb5eff50`
- `research/intraday-vol-entry-2026-10@19768051`
- `research/crowding-vol-timing-2026-10@8a0f5b51`
- `research/option-edge-2026-10@78f28c5a`

时间和主研究链上最新的是 `research/crowding-vol-timing-2026-10`。本分支以它为基准创建。

PR #1 是 `research/layered-2026-10 → main`，仍为 open、可合并，但没有 review 或评论。duration、intraday-vol、crowding 三个后续分支不在 PR #1 head 中，所以仅阅读 PR #1 不能得到当前 v5 设计。

## quant-handbook 对照

[QuantMLResearch/quant-handbook](https://github.com/QuantMLResearch/quant-handbook) 没有分钟 Noise Band 实现。可复用的方法是：

- `examples/factor_utils.py` 的 HAC 均值检验思想；
- `examples/lecture_18_financial_cv_purging_embargo.py` 的 forward-only、purging 和日期分组原则；
- `examples/lecture_02_backtest_sanitizer.py` 对 DSR、试验次数和数据窥探的限制说明。

不应直接迁移 A 股横截面 IC、Top 名单换手或合成示例。它们和单 ETF 日内状态机不是同一个统计问题。手册也没有 block bootstrap、SPA、Reality Check 或完整 PBO 实现。

## 新增实验

新增 `studies/noise_band_validation.py`，实现：

1. raw 与 gap-anchored coverage；
2. 按年份、时点、QQQ/SPY 的覆盖稳定性；
3. 带外事件的 5 分钟、30 分钟、MOC 延续收益；
4. 控制连续位移、s3、gap 和 clock fixed effects 后的增量阈值回归，标准误按交易日聚类；
5. 原论文、当前 s3、q60-L60、q60-L60+s3 四个固定 band；四个版本全部同次重跑，避免旧 cache provenance 不明；
6. 所有候选重跑 30 phase、FOMC 路径和下游仓位训练，再进入 PIT-corrected v5 组合；
7. 15:40 持仓按 15:41 open 重建退出成本；cap 使用开盘前已知的 `k=2` 上界，避免当日 10:00 信息回写核心仓；
8. HAC、Holm、stationary bootstrap block 5/10/20 和同步 max-t family-wise 检验；
9. 历史样本固定截止 2026-10-30，forward 从 2026-11-02 开始；不足 252 日时禁止给出采用判断；
10. 数据 MD5、参数、分支、冻结日期和随机种子的元数据记录。

q60-L60 是本轮唯一新增的 band 估计思路。它不是因为历史 Sharpe 最优而选，而是为了给 noise range 一个明确的 60% raw coverage 目标。`q60-L60+s3` 单独检验现有 vol/volume 条件状态在已校准基线之上是否仍有增量。

## 执行状态

已完成：

- Python 语法编译；
- Ruff 静态检查；
- 完整 `requirements.txt` 下的策略模块导入；
- 390 根分钟 bar 的合成数据 smoke path；
- point-in-time profile、coverage、continuation、HAC、stationary bootstrap 和 max-t 的运行检查。

未执行真实市场数据实验。当前沙盒没有 `raw/min1/QQQ.csv`、`raw/min1/SPY.csv` 及其余约 1.2GB 原始数据，`ETFT_DATA` 也未设置。仓库只提交了 MD5 manifest 和聚合结果，没有逐日 pickle，无法从 CSV 汇总表反推出 coverage 或配对收益序列。

运行命令：

```bash
export ETFT_DATA=/path/to/data
python studies/noise_band_validation.py --section all --bootstrap 2000
```

在真实数据输出产生前，不应声称 q60 改善了策略，也不应把当前 s3 判为失败。当前可支持的结论仅是：**s3 的历史点估计略正，但置信区间宽、左尾变差，尚未证实；原 band 还没有 noise-range 校准证据。**
