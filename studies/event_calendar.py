"""确定性事件日历(事前已知)。仅由日历规则决定,不使用任何价格。
FOMC 决议日 = 计划内会议的第二天(14:00 ET 公告)。2021–2026 来自 federalreserve.gov/monetarypolicy/fomccalendars.htm(2026-10-11 抓取);
2019–2020 为已公布的计划内会议(不含 2020-03 紧急会议)。"""
from __future__ import annotations
import pandas as pd

FOMC_DECISION = pd.to_datetime([
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (计划内)
    "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
])


def event_frame(days: pd.DatetimeIndex, last_idx) -> pd.DataFrame:
    days = pd.DatetimeIndex(days); pos = {d: i for i, d in enumerate(days)}
    E = pd.DataFrame(0.0, index=days, columns=["e_fomc", "e_fomc_eve", "e_fomc_post", "e_opex", "e_quad", "e_month_end",
                                              "e_month_start", "e_nfp", "e_pre_holiday", "e_post_holiday", "e_half"])
    for d in FOMC_DECISION:
        if d in pos:
            i = pos[d]; E.iloc[i, 0] = 1.0
            if i > 0: E.iloc[i - 1, 1] = 1.0
            if i + 1 < len(days): E.iloc[i + 1, 2] = 1.0
    ym = pd.Series(days, index=days).groupby([days.year, days.month])
    for (y, m), g in ym:
        g = pd.DatetimeIndex(g.values)
        fr = [d for d in pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D") if d.month == m and d.weekday() == 4]
        if len(fr) >= 3:
            third = fr[2]; cand = g[g <= third]
            if len(cand):
                d = cand[-1]; E.loc[d, "e_opex"] = 1.0
                if m in (3, 6, 9, 12): E.loc[d, "e_quad"] = 1.0
        if len(fr) >= 1:
            f1 = fr[0]
            if f1 in E.index: E.loc[f1, "e_nfp"] = 1.0
        E.loc[g[-2:], "e_month_end"] = 1.0
        E.loc[g[0], "e_month_start"] = 1.0
    bd = pd.bdate_range(days[0] - pd.Timedelta(days=7), days[-1] + pd.Timedelta(days=7))
    missing = set(bd) - set(days)
    for d in days:
        prv = d - pd.offsets.BDay(1); nxt = d + pd.offsets.BDay(1)
        if nxt in missing: E.loc[d, "e_pre_holiday"] = 1.0
        if prv in missing: E.loc[d, "e_post_holiday"] = 1.0
    E["e_half"] = (pd.Series(last_idx, index=days).to_numpy() < 389).astype(float)
    return E
