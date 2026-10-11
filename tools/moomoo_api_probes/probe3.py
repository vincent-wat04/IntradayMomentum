"""探测 3：复权/股息接口，以及 request_history_kline 的 autype 参数。"""
import inspect

import pandas as pd
from moomoo import OpenQuoteContext, RET_OK, KLType, AuType

pd.set_option("display.width", 240)
q = OpenQuoteContext(host="127.0.0.1", port=11111)

print("request_history_kline 签名:")
print(" ", inspect.signature(q.request_history_kline), flush=True)

print("\nAuType 可选值:", [x for x in dir(AuType) if not x.startswith("_")], flush=True)

for nm in ("get_rehab", "get_corporate_actions_dividends", "get_dividend_calendar"):
    fn = getattr(q, nm, None)
    print(f"\n--- {nm}: {'有' if fn else '无'}")
    if fn:
        try:
            print("  签名:", inspect.signature(fn), flush=True)
        except Exception as ex:
            print("  签名读取失败", ex, flush=True)

r, d = q.get_rehab("US.QQQ")
print("\nget_rehab(US.QQQ):", "FAIL " + str(d) if r != RET_OK else f"{len(d)} 行")
if r == RET_OK and len(d):
    print(d.tail(6).to_string(index=False), flush=True)
    print("列:", list(d.columns), flush=True)

# 不复权 vs 前复权 的同一天对比，确认 autype 生效
for au, lab in ((AuType.NONE, "NONE 不复权"), (AuType.QFQ, "QFQ 前复权")):
    out = q.request_history_kline("US.QQQ", start="2026-09-18", end="2026-09-18",
                                  ktype=KLType.K_1M, max_count=3, autype=au)
    r2, d2 = out[0], out[1]
    print(f"\n{lab}: ", "FAIL " + str(d2) if r2 != RET_OK else
          d2[["time_key", "open", "close"]].head(2).to_string(index=False), flush=True)

q.close()
print("\nOK", flush=True)
