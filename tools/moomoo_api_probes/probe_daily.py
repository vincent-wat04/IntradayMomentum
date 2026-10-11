"""探测：不走历史 K 线额度，能不能拿到足够长的 QQQ 日线？

get_cur_kline 是**实时订阅**接口，不属于"历史 K 线额度"（300 个标的/7 天滚动）。
本探针只用来确认：K_DAY 最多能返回多少根、日期是否连续、是否含当天未完成的 bar。
"""
import sys
import time

from moomoo import AuType, KLType, OpenQuoteContext, RET_OK, SubType

q = OpenQuoteContext(host="127.0.0.1", port=11111)
code = sys.argv[1] if len(sys.argv) > 1 else "US.QQQ"
ret, data = q.get_history_kl_quota(get_detail=False)
print("额度(前):", data)
r, msg = q.subscribe([code], [SubType.K_DAY])
print("subscribe:", r, msg if r != RET_OK else "ok")
time.sleep(2)
r, d = q.get_cur_kline(code, num=1000, ktype=KLType.K_DAY, autype=AuType.QFQ)
print("get_cur_kline ret:", r)
if r == RET_OK:
    print("根数:", len(d), " 列:", list(d.columns))
    print("首:", d.time_key.iloc[0], "尾:", d.time_key.iloc[-1], " 最后收盘:", d.close.iloc[-1])
    d[["time_key", "open", "high", "low", "close", "volume"]].to_csv("/tmp/qqq_daily_probe.csv", index=False)
else:
    print(d)
ret, data = q.get_history_kl_quota(get_detail=False)
print("额度(后):", data)
q.close()
