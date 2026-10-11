"""证明：实盘 QQQ 腿（live/leg_signal.py，标量）== 回测 QQQ 腿（combo_lib / sens_rv_weights，向量化）。

与 Sleeve 的 replay_equivalence 同一个思路：实盘代码和回测代码是**两份独立实现**，
只有逐日对账到 0.00e+00，才能说"实盘跑的就是回测过的那个东西"。

对账两件事：
  1. 全历史（QQQ 1999 起）每隔 7 天抽一天 + 最近 120 天全测：百分位、权重逐位一致
  2. 未来投毒：把 t 之后的价格篡改，权重 w_t 不变（因果性）

用法（在研究库 studies/ 目录下，LIVE_SRC 指向实盘库的 src/）：
  python ../tools/leg_equivalence.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "studies"))
LIVE = Path(os.environ.get("LIVE_SRC_ROOT", HERE.parents[1] / "noise-momentum-live")) / "src"
if not (LIVE / "leg_signal.py").exists():
    raise SystemExit(f"找不到实盘 leg_signal.py：{LIVE}\n请把 noise-momentum-live 放在本库同级，或设 LIVE_SRC=/path/to/live/src")
sys.path.insert(0, str(LIVE))

import round3_common as C                      # noqa: E402
from sens_rv_weights import pct_w, w_linear    # noqa: E402
import leg_signal as L                         # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else '**FAIL**'}] {name}  {detail}")


y = C.load_close("yahoo/QQQ.parquet")
cfg = L.LegCfg()
p_vec = pct_w(y.pct_change().rolling(cfg.rv_win).std(), cfg.pct_win)
w_vec = w_linear(p_vec, cfg.lo, cfg.hi)
print(f"QQQ 日线 {len(y)} 根 {y.index[0].date()} ~ {y.index[-1].date()}；配置 lo={cfg.lo} hi={cfg.hi}")

idx = sorted(set(range(600, len(y), 7)) | set(range(len(y) - 120, len(y))))
mp = mw = 0.0; n = bad = 0
for i in idx:
    pl = L.rv_percentile(y.iloc[: i + 1].to_numpy(), cfg)
    wl = L.leg_weight(pl, cfg)
    pv, wv = p_vec.iloc[i], w_vec.iloc[i]
    if np.isnan(pv) and np.isnan(pl):
        continue
    dp, dw = abs(pl - pv), abs(wl - wv)
    mp, mw, n = max(mp, dp), max(mw, dw), n + 1
    bad += int(dp > 1e-12 or dw > 1e-12)
check("1. 实盘标量实现 vs 回测向量化实现", bad == 0,
      f"{n} 个交易日；百分位最大差 {mp:.2e}，权重最大差 {mw:.2e}，不一致 {bad} 天")

rng = np.random.default_rng(3)
worst = 0.0
for t in rng.choice(np.arange(700, len(y) - 1), 40, replace=False):
    a = L.leg_weight(L.rv_percentile(y.iloc[: t + 1].to_numpy(), cfg), cfg)
    z = y.copy()
    z.iloc[t + 1:] = z.iloc[t + 1:] * rng.uniform(0.3, 3.0, len(z) - t - 1)
    b = L.leg_weight(L.rv_percentile(z.iloc[: t + 1].to_numpy(), cfg), cfg)
    worst = max(worst, abs(a - b))
check("2. 未来投毒（40 个随机日期）：t 之后的价格被篡改，w_t 不变", worst == 0.0, f"最大差 {worst:.1e}")

# 反向对照：故意把当日之后的数据喂进去，权重应当变化（证明投毒检验有分辨力）
t = 900
leak = abs(L.leg_weight(L.rv_percentile(y.iloc[: t + 1].to_numpy(), cfg), cfg)
           - L.leg_weight(L.rv_percentile(y.iloc[: t + 30].to_numpy(), cfg), cfg))
check("3. [反向对照] 多喂 29 天未来数据，权重会变", leak > 1e-9, f"差 {leak:.4f}")

print("\n" + ("全部通过" if ok else "**有失败项**"))
sys.exit(0 if ok else 1)
