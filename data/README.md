# 数据

原始数据约 1.2GB，**不随库交付**。用环境变量指向已有目录即可：

```bash
export ETFT_DATA=/path/to/data      # 内含 raw/min1、raw/yahoo、raw/cboe ...
```

```
raw/min1/{QQQ,SPY,...}.csv      美股 1 分钟 bar，time_key = bar 结束时刻（moomoo 口径）
raw/yahoo/{QQQ,IDX_GSPC,IDX_NDX,...}.parquet   日频复权价
raw/cboe/{VIX,VXN,VVIX,COR1M,COR3M,...}.parquet
raw/squeeze_dix_gex.parquet          SqueezeMetrics DIX/GEX（每日标量）
```
