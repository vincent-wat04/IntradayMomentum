"""按 PROPOSAL_slowbear.md 第 3 节的冻结规则自动判定（读 outputs/slowbear_study.csv，不重新挑选）。

解释（跑结果前就这样定义，这里写明以免事后歧义）：
  (a) 在 SPX、NDX 上，互联网泡沫与金融危机的"峰→谷"都严格优于 H0
  (b) 全样本 MDD 优于 H0 且 Calmar ≥ H0 —— SPX、NDX 都要满足
  (c) 全样本 ΔSR(对照) ≥ 0 —— SPX、NDX 都要满足
  (d) 2010-2026 夏普 ≥ H0 − 0.10 —— SPX、NDX 都要满足
  (e) RUT、NYA 上 (a)(b) 同向：这四个市场里 (a)(b) 合计至少 3/4 个市场成立
"""
from pathlib import Path

import pandas as pd

D = pd.read_csv(Path(__file__).resolve().parent / "outputs" / "full" / "intraday" / "slowbear_study.csv")
RULES = [r for r in D.rule.unique() if r not in ("H0 浅蓝", "B&H")]


def full(m, r):
    x = D[(D.mkt == m) & (D.rule == r) & D.episode.isna()].iloc[0]
    return x


def epi(m, r, e):
    return D[(D.mkt == m) & (D.rule == r) & (D.episode == e)].iloc[0].down


def ok_a(m, r):
    return all(epi(m, r, e) > epi(m, "H0 浅蓝", e) + 1e-9 for e in ("互联网泡沫", "金融危机"))


def ok_b(m, r):
    a, b = full(m, r), full(m, "H0 浅蓝")
    return a.mdd > b.mdd and a.calmar >= b.calmar - 1e-9


out = []
print(f"{'方案':18s}{'(a)':>6}{'(b)':>6}{'(c)':>6}{'(d)':>6}{'(e)':>6}   总判定    各项细节")
for r in RULES:
    a = all(ok_a(m, r) for m in ("SPX", "NDX"))
    b = all(ok_b(m, r) for m in ("SPX", "NDX"))
    c = all(full(m, r).dsr >= 0 for m in ("SPX", "NDX"))
    d = all(full(m, r).sr_post2010 >= full(m, "H0 浅蓝").sr_post2010 - 0.10 for m in ("SPX", "NDX"))
    ab4 = sum(ok_a(m, r) and ok_b(m, r) for m in ("SPX", "NDX", "RUT", "NYA"))
    e = ab4 >= 3
    allok = a and b and c and d and e
    print(f"{r:18s}{str(a):>6}{str(b):>6}{str(c):>6}{str(d):>6}{str(e):>6}   {'通过' if allok else '否决':6s}   "
          f"(a)(b)同时成立的市场数={ab4}/4")
    out.append(dict(rule=r, a=a, b=b, c=c, d=d, e=e, verdict=allok))
pd.DataFrame(out).to_csv(Path(__file__).resolve().parent / "outputs" / "full" / "intraday" / "slowbear_verdict.csv", index=False)
