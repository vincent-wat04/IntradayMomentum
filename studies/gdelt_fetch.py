"""GDELT DOC 2.0 日度新闻量/语调(聚合,无原文)。限流 1 请求/5 秒。输出 ETFT_DATA/raw/gdelt/*.parquet
volume = 该主题文章占当日全部收录文章的比例('Volume Intensity');tone = 平均语调。"""
import requests, time, json, urllib.parse, sys, os
import pandas as pd
from pathlib import Path
OUT = Path(os.environ["ETFT_DATA"]) / "raw" / "gdelt"; OUT.mkdir(parents=True, exist_ok=True)
TOPICS = {
 "fed": '"federal reserve" OR "FOMC" OR "interest rate" OR "rate hike" OR "rate cut"',
 "inflation": 'inflation OR CPI OR "consumer prices"',
 "jobs": '"jobs report" OR payrolls OR unemployment',
 "stocks": '"stock market" OR "Wall Street" OR Nasdaq OR "S&P 500"',
 "crash": 'selloff OR plunge OR crash OR "market turmoil" OR "volatility"',
 "war": 'war OR missile OR strike OR invasion OR ceasefire',
 "tariff": 'tariff OR sanctions OR "trade war" OR "trade deal"',
 "bank": '"bank run" OR "bank failure" OR default OR "credit crisis"',
 "ai_tech": '"artificial intelligence" OR semiconductor OR Nvidia OR "big tech"',
}
def fetch(query, mode, s, e):
    u = ("https://api.gdeltproject.org/api/v2/doc/doc?query=" + urllib.parse.quote(f"({query}) sourcelang:eng") + f"&mode={mode}&format=json&startdatetime={s}&enddatetime={e}&timelinesmooth=0")
    for att in range(6):
        time.sleep(5.5)
        r = requests.get(u, timeout=120)
        if r.status_code == 200 and r.text.strip().startswith("{"):
            try:
                d = r.json()["timeline"][0]["data"]; return pd.Series({pd.to_datetime(x["date"]): x["value"] for x in d})
            except Exception: return pd.Series(dtype=float)
        time.sleep(8 * (att + 1))
    return pd.Series(dtype=float)
if __name__ == "__main__":
    spans = [("20190101000000", "20201231235959"), ("20210101000000", "20221231235959"), ("20230101000000", "20241231235959"), ("20250101000000", "20261005235959")]
    for name, q in TOPICS.items():
        f = OUT / f"{name}.parquet"
        if f.exists(): continue
        parts = {}
        for mode in ("timelinevolraw", "timelinetone"):
            pieces = [fetch(q, mode, s, e) for s, e in spans]
            parts[mode] = pd.concat([p for p in pieces if len(p)])
        df = pd.DataFrame({"vol": parts["timelinevolraw"], "tone": parts["timelinetone"]}); df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_parquet(f); print(name, len(df), df.index.min(), df.index.max(), flush=True)
