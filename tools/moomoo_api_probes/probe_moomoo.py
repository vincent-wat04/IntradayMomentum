"""分步探测 moomoo 链路：行情（已知可用）-> 交易上下文 -> 模拟盘账户 -> 下单权限。

每一步都 flush 并打印耗时，这样挂住时能准确定位在哪一步。
在服务器上跑：/opt/qmacro/.venv/bin/python -u /opt/live_noise/probe_moomoo.py
"""
import sys
import time
import traceback

import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
HOST, PORT = "127.0.0.1", 11111


def step(name, fn):
    t0 = time.time()
    print(f"\n>>> {name} ...", flush=True)
    try:
        r = fn()
        print(f"<<< {name} 完成 {time.time()-t0:.1f}s", flush=True)
        return r
    except Exception:
        print(f"<<< {name} 异常 {time.time()-t0:.1f}s", flush=True)
        traceback.print_exc()
        return None


def main():
    from moomoo import (OpenQuoteContext, OpenSecTradeContext, RET_OK, TrdMarket,
                        TrdEnv, SecurityFirm, KLType, Currency, TradeDateMarket)
    import moomoo
    print("moomoo", moomoo.__version__, flush=True)

    # ---------- 1. 行情（已验证可用）
    q = step("建立行情连接 OpenQuoteContext", lambda: OpenQuoteContext(host=HOST, port=PORT))
    if q is None:
        return

    def snap():
        r, d = q.get_market_snapshot(["US.QQQ", "US.TQQQ"])
        if r != RET_OK:
            print("  失败:", d, flush=True); return None
        print(d[["code", "last_price", "prev_close_price", "open_price", "update_time",
                 "lot_size"]].to_string(index=False), flush=True)
        return d
    step("快照 US.QQQ / US.TQQQ", snap)

    def kline():
        r, d = q.get_history_kline("US.QQQ", start="2026-09-25", end="2026-10-02",
                                   ktype=KLType.K_1M, max_count=3)
        print(("  " + str(d)[:400]) if r != RET_OK else d.to_string(index=False), flush=True)
        return d
    step("历史 1 分钟 K 线（额度敏感）", kline)

    def tdays():
        r, d = q.get_trading_days(market=TradeDateMarket.US, start="2026-10-01", end="2026-10-15")
        print("  ", str(d)[:300], flush=True)
        return d
    step("美股交易日历", tdays)

    # ---------- 2. 交易上下文：逐个 security_firm 试
    for firm_name in ("FUTUSECURITIES", "FUTUINC", "MOOMOOSECURITIES"):
        firm = getattr(SecurityFirm, firm_name, None)
        if firm is None:
            print(f"\n  SDK 无 SecurityFirm.{firm_name}，跳过", flush=True)
            continue

        def mk(firm=firm):
            return OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host=HOST, port=PORT,
                                       security_firm=firm)
        trd = step(f"建立交易连接 security_firm={firm_name}", mk)
        if trd is None:
            continue

        def accs():
            r, d = trd.get_acc_list()
            if r != RET_OK:
                print("  get_acc_list 失败:", d, flush=True); return None
            print(d.to_string(index=False), flush=True)
            return d
        d = step(f"[{firm_name}] 账户列表", accs)
        if d is not None and len(d):
            sim = d[d.trd_env == "SIMULATE"] if "trd_env" in d.columns else d
            print(f"  模拟盘账户 {len(sim)} 个", flush=True)
            for aid in sim.acc_id.tolist()[:3]:
                def info(aid=aid):
                    r, f = trd.accinfo_query(acc_id=aid, trd_env=TrdEnv.SIMULATE,
                                             currency=Currency.USD)
                    print("  ", str(f)[:500] if r != RET_OK else
                          f[[c for c in ("power", "total_assets", "cash", "market_val",
                                         "available_funds") if c in f.columns]]
                          .to_string(index=False), flush=True)
                    return f
                step(f"[{firm_name}] 资金 acc_id={aid}", info)

                def pos(aid=aid):
                    r, p = trd.position_list_query(acc_id=aid, trd_env=TrdEnv.SIMULATE)
                    print("   持仓数:", len(p) if r == RET_OK else p, flush=True)
                    return p
                step(f"[{firm_name}] 持仓 acc_id={aid}", pos)
        try:
            trd.close()
        except Exception:
            pass
        if d is not None and len(d):
            break

    q.close()
    print("\n探测结束", flush=True)


if __name__ == "__main__":
    main()
