> **历史文档，已被 noise-momentum-live/README.md 取代。** 保留它是为了记录当时的设计推理。

# 实盘系统：日内噪声区域策略（QQQ 信号 → TQQQ 执行，moomoo 模拟盘）

已部署并在跑。服务器 `<SERVER_IP>`，代码在 `/opt/live_noise/`，
systemd 服务 `live-noise.service`，账户 **acc_id 4813475（SIMULATE）**。

## 1. 为什么是这个策略、这个配置

PART8/PART9 的结论：这是整个项目 9 个 PART 里**唯一在 beta 对冲后还站得住**的东西
（beta 0.002、与买入持有相关 0.003、年化 alpha 19.3%、t=2.66）。

配置就是 PART9 §1.3 的 B 档 + §4.3 的实盘修正：

| 项 | 取值 | 依据 |
|---|---|---|
| 信号标的 | QQQ | SPY 已衰减（2026 年 Sharpe −1.42，PART8 §4.5） |
| 噪声带宽 vm | 1.0 | 论文原值 |
| σ 回看 | 14 个交易日 | 论文原值 |
| 检查时点 | 10:00, 10:30, ..., 15:00（bar **结束**时刻） | 论文原值 |
| 止损 | 多头 max(UB, 当日累计VWAP)、空头 min(LB, VWAP) | 论文的改进版，E19 实测优于 band 止损 |
| 仓位 | min(4.0, 2% / σ_daily_14) | 论文的 vol targeting；PART9 §1.3 实测 Sharpe 1.250 → **1.375**，beta 0.043 → **0.002** |
| 止损/重入语义 | **hold** | AUDIT §2.4：三种语义 Sharpe 1.122/1.117/**1.147** |
| **执行标的** | **TQQQ**（beta 3.0） | PART9 §1.5 + 实测购买力限制，见 §4 |
| 隔夜 | **0**（15:55 强平） | 策略定义如此；也因此不吃 TQQQ 的每日再平衡衰减、不付隔夜利息 |
| 不采用 | 成交量调整、GEX 缩放、波动率条件化 | PART9 §1.3 / §2.4：test 期都不支持，置换检验 p=0.092 |

**预期**（PART9，TQQQ 成本口径，test 2023-2026）：Sharpe 1.138、年化 16.3%、回撤 −15.0%。
**但前瞻预期要打折**：2025 年 Sharpe 0.56、2026 年 0.33（PART9 §1.6）。按 0.6~1.0 做计划。

## 2. 架构：回测与实盘共用唯一一份决策代码

本项目反复出现的问题是"实现漂移"——同一个策略写两遍，两遍不一样。所以：

```
live/strategy_signal.py   decide() 纯函数（无 I/O、无全局状态、无 pandas）
        │
        ├── live/replay.py     用历史 bar 逐根回放 ──► 必须逐位等于已审计的回测
        └── live/runner.py     用实时 bar 调用     ──► 实盘
```

`decide(plan, clock, close, vwap, pos, cfg)` 返回**目标仓位**。它只接受
`reentry="hold"`，另外两种语义会**显式报错**而不是静默走偏：
`same_bar`（同价平掉再原样重开）在"返回目标仓位"的架构下结构上不可表达
——而这正是我们想要的，因为它比 hold 差 0.025 Sharpe。

**等价性证据**（`python live/replay.py`）：

```
[PASS] sizing=fixed     reentry=hold   最大差 0.00e+00  不一致 0 天  累计 0.718434 / 0.718434
[PASS] sizing=voltarget reentry=hold   最大差 0.00e+00  不一致 0 天  累计 1.230664 / 1.230664
[PASS] signal.sigma_profile 与 pandas 版本一致（40 天 × 3 时点，不一致 0 处）
```

链条：`decide()` 回放 == `audit/reimpl_noise.py` == 被测引擎（逐位，见 AUDIT §2.4）。
**任何对策略的修改都必须重跑 replay.py。**

## 3. 实盘状态 vs 回测状态 的对账

实盘的 σ/敞口来自服务器本地 1 分钟缓存（moomoo 历史灌入 + 实时流），
回测来自 etf_timing 的 min1 面板。两条路径完全独立。`python live/crosscheck.py`：

| 对账项 | 结果 |
|---|---|
| bar 收盘价（165 个点、33 天重叠） | 最大相对差 **0.00e+00** |
| σ profile（19 天 × 6 时点） | 最大相对差 **0.00e+00** |
| 敞口（18 天） | 最大相对差 **1.1e-05** |

这个对账过程查出了两个**只存在于实盘路径**的错误（复权口径、除息日带参考价），
已修复，详见 AUDIT §4。

## 4. 为什么用 TQQQ 执行 —— 被购买力实测逼出来的

SOTA 的目标敞口实测均值 **1.95x**。模拟盘账户：现金 $996,248、显示买力 $1,992,496（2:1）。

实测结果：**下单 2500 股 QQQ（名义 $1.886M）被拒绝 —— "购买力不足"**，
尽管名义金额低于显示的买力。

走 TQQQ 则毫无压力：1.95x QQQ 等价敞口 = TQQQ 名义 **0.65x** 资金 = $646k。
实际成交（2026-10-05 13:30）：

```
decision: clock=13:30 close=754.635 target=1.9463 target_shares=7821 trade_px=82.646
          why=突破上轨开多 (close=754.6350, UB=753.2555, LB=745.8150, e=1.946)
order_final: order_id=3300779 status=FILLED_ALL dealt_qty=7821 avg_price=82.65
```

7821 × $82.65 = **$646,406** = $1.939M / 3 ✓

> 这正是 PART9 §1.5 的结论被实证：**TQQQ 的价值不是成本更低（它贵 2.7 倍），
> 而是用 1/3 的名义金额拿到同样敞口**，而账户的购买力约束是绑在名义金额上的。

仓位换算（`runner.py`）：
```
目标 TQQQ 股数 = 目标敞口 × 资金 / (TQQQ 价格 × trade_beta)
当前敞口比例  = TQQQ 持股 × TQQQ 价格 × trade_beta / 资金
```
`trade_beta=3.0`（PART2 实测 TQQQ 日频 realized beta 2.96、相关 0.9991）。

## 5. 实测交易成本 —— 回测里最后一个假设值，现在有实测了

回测用的是 `(半价差 0.005 + 佣金 0.0045)/价格`，这是**假设值**，而策略的成本
盈亏平衡只有 2.55bp/边（PART8 §4.4）。所以必须实测。

做法（`live/measure_cost.py`）：真实规模下单、立刻平回，
`实际成本 = 价格损益 − 权益变化`。

| 规模 | 买入均价 | 卖出均价 | 推算成本（两腿） | 单边 | 每股/边 |
|---|---|---|---|---|---|
| 1 股 ×4 腿 | — | — | $3.98 | — | ~$1.00（**每单最低佣金**，真实规模下消失） |
| 100 股 | 754.45 | 754.485 | $1.997 | **0.1323 bp** | **$0.009985** |

> **回测假设 $0.0095/股/边，实测 $0.009985/股/边 —— 吻合，偏差 5%。成本模型验证通过。**
>
> 两个必须说清的保留：
> 1. 模拟盘**按最新价成交，不建模买卖价差**（实测里卖出价甚至高于买入价）。
>    所以这 $0.01/股是**纯佣金**。真实世界还要加半价差（$0.005/股 = 0.066bp @$754）。
> 2. TQQQ 股价只有 $82.65，每股固定成本对应的相对成本是 QQQ 的 9 倍
>    （$0.01/82.65 = 1.21bp/边 vs QQQ 0.13bp）。但因为只交易 1/3 的名义金额，
>    **单位敞口成本 = 1.21/3 = 0.40bp**，仍远低于 2.55bp 的盈亏平衡。
>    PART9 §1.2 用的 0.526bp 是更保守的估计。

## 6. 安全栏

| 栏目 | 设置 | 代码位置 |
|---|---|---|
| **只允许模拟盘** | `ALLOW_REAL_MONEY=False` **且** 需命令行 `--i-understand-real-money`，缺一不可 | `broker.py` |
| 持仓以券商为准 | 每次决策前 `position_list_query`，不依赖进程内部推测 → 重启/部分成交/手工干预都能自愈 | `broker.py:position_shares` |
| 跨零反手拆单 | 先平后开两条腿（已在模拟盘验证：腿数 2，平仓 SELL1@754.57 + 反向开仓 SELL1@754.68） | `broker.py:go_to_shares` |
| 幂等 | 每个检查时点只决策一次，写入 `state/day_*.json`；目标==当前时不下单 | `runner.py` |
| 行情过期 | bar 超过 120 秒未更新 → 跳过本次决策 | `Limits.bar_staleness_sec` |
| 市场状态 | 非 NORMAL/TRADING/MORNING/AFTERNOON → 跳过 | `runner.py` |
| 单日下单上限 | 12 单 | `Limits.max_orders_per_day` |
| 敞口上限 | 4.0x（截断 + 告警） | `Limits.max_exposure` |
| **单日亏损熔断** | 权益回撤 5% → 清仓并停止当日（平仓逻辑仍执行） | `Limits.daily_loss_stop` |
| **强制平仓** | 15:55 无条件平仓，隔夜敞口严格 0 | `Limits.flatten_clock` |
| 时钟来源 | 用 bar 的 `time_key`（交易所时间），**不信本机时钟**（服务器是 CST，差 12 小时） | `runner.py:ny_now` |
| 内存上限 | `MemoryMax=220M`、`CPUQuota=40%`（这台机器 `svt-serve` 占 2.4G，只剩 ~225M） | systemd unit |
| 审计日志 | 每个决策/订单/成交写 `state/audit.jsonl`，字段足以还原当时依据 | `broker.py:log` |

## 7. 运维

```bash
# 启动前自检（14 项，不通过就别启动）
/opt/qmacro/.venv/bin/python -u /opt/live_noise/preflight.py US.QQQ

# 服务
systemctl status  live-noise
systemctl restart live-noise
systemctl stop    live-noise          # 注意：停服务不会自动平仓，先确认持仓为 0
journalctl -u live-noise -n 50

# 日志
tail -f /opt/live_noise/state/runner.log
tail -5  /opt/live_noise/state/audit.jsonl | python -m json.tool

# 首次/缓存损坏时重灌 1 分钟历史（会消耗 K 线额度；QQQ 已在额度集合内所以免费）
/opt/qmacro/.venv/bin/python -u /opt/live_noise/runner.py \
    --code US.QQQ --dry-run --bootstrap-days 25 --once

# 下单链路验证（1 股走完 开多→反手→平回）
/opt/qmacro/.venv/bin/python -u /opt/live_noise/test_order_path.py

# 成本实测
/opt/qmacro/.venv/bin/python -u /opt/live_noise/measure_cost.py 100
```

**K 线额度注意**：`get_history_kl_quota` 显示 **297/300 已用**（7 天滚动，主要被
`/opt/longvol` 的期权扫描消耗）。QQQ/TQQQ 已在已用集合内，重复拉取不占新额度；
**不要用这套代码去拉新标的的历史**。

## 8. 走真实资金之前必须做的三件事

1. **用真实账户的小额成交重测成本。** 模拟盘不建模买卖价差，而 TQQQ 的日内波动是
   QQQ 的 3 倍，做市商会要更宽的价差。盈亏平衡 2.55bp，现在估的是 0.40bp，
   安全边际看起来有 6 倍 —— 但这个 6 倍完全建立在"价差 = 1 美分"的假设上。
2. **跑满至少一个完整月的模拟盘**，把每日 `audit.jsonl` 的决策与离线回放逐笔比对
   （`replay.py` 的逻辑可以直接喂当日 bar 做比对），确认没有实盘特有的偏差。
3. **接受前瞻预期是 0.6~1.0 而不是 1.14**（PART9 §1.6 的逐年衰减），
   并设好停止规则（建议：滚动 12 个月 Sharpe < 0 则停）。

## 9. 文件清单

```
live/
  strategy_signal.py     decide() 纯函数 + σ/日波动/敞口的 PIT 计算   ← 唯一的决策代码
  replay.py              逐根回放证明 decide() == 已审计回测
  crosscheck.py           实盘状态 vs 回测状态对账（--pull 拉缓存）
  broker.py              moomoo 封装 + 安全栏 + 审计日志
  runner.py              主循环（建计划 → 检查时点决策 → 15:55 强平）
  preflight.py           启动前自检 14 项（含 7 个场景的决策真值表）
  test_order_path.py     下单链路验证（开多 → 跨零反手 → 平回 → 幂等）
  measure_cost.py        真实规模下的成本实测
  probe_moomoo.py        分步链路探测（排障用）
  live-noise.service     systemd 单元
  sshx.py                服务器连接助手（带重试，这台机器 SSH banner 常超时）
audit/
  harness.py             投毒/比较/绩效 工具箱
  reimpl_noise.py        独立重写的显式账本实现（三种语义 × 两种成本口径）
  audit_engine.py        引擎层 19 项
  audit_data.py          数据层 20 项
  audit_factors.py       因子/横截面层 16 项
```
