"""探测 2：额度、市场状态、交易日历、历史 K 线、订阅、实时 K 线。"""
import pandas as pd
from moomoo import (OpenQuoteContext, RET_OK, KLType, SubType, TradeDateMarket)

pd.set_option("display.width", 220)
q = OpenQuoteContext(host="127.0.0.1", port=11111)

r, d = q.get_history_kl_quota(get_detail=False)
print("历史K线额度:", d, flush=True)

r, d = q.get_market_state(["US.QQQ", "US.TQQQ"])
print("\n市场状态:"); print(d if r != RET_OK else d.to_string(index=False), flush=True)

r, d = q.request_trading_days(market=TradeDateMarket.US, start="2026-09-20", end="2026-10-16")
print("\n交易日:", d if r != RET_OK else [x["time"] for x in d], flush=True)

out = q.request_history_kline("US.QQQ", start="2026-10-02", end="2026-10-02",
                              ktype=KLType.K_1M, max_count=1000)
r, d = out[0], out[1]
print("\n历史1分钟 2026-10-02:", "FAILED " + str(d) if r != RET_OK else
      f"{len(d)} 根, 首 {d.time_key.iloc[0]} 末 {d.time_key.iloc[-1]}", flush=True)
if r == RET_OK:
    print(d.head(3).to_string(index=False), flush=True)
    print(d.tail(2).to_string(index=False), flush=True)
    print("列:", list(d.columns), flush=True)
r2, q2 = q.get_history_kl_quota(get_detail=True)
print("\n拉取后额度:", q2, flush=True)

r, d = q.subscribe(["US.QQQ"], [SubType.K_1M, SubType.QUOTE])
print("\n订阅:", r, d, flush=True)

r, d = q.get_cur_kline("US.QQQ", num=3, ktype=KLType.K_1M)
print("\n实时1分钟:"); print(d if r != RET_OK else d.to_string(index=False), flush=True)

r, d = q.get_market_snapshot(["US.QQQ"])
if r == RET_OK:
    cols = [c for c in ("code", "last_price", "open_price", "prev_close_price", "update_time",
                        "sec_status", "lot_size") if c in d.columns]
    print("\n快照:"); print(d[cols].to_string(index=False), flush=True)

q.close()
print("\nOK", flush=True)
