# quote_features/spread.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteSpreadOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    只读 ['symbol','ask_price','bid_price','participant_timestamp']，单次流式扫描整天，
    计算按事件简单平均的日内相对价差： 2*(ask - bid) / (ask + bid)，
    仅统计 09:30–16:00 ET（RTH），并将交叉报价(ask<bid)截断为 0。
    输出：['symbol','value'] 按 PV 样本顺序对齐。
    """
    name = "quote_spread_all"
    description = "RTH mean of 2*(ask-bid)/(ask+bid) per symbol (onefile, single pass)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    # 配置（如无特殊需要，不必改）
    RTH_START = dt.time(9, 30)
    RTH_END   = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    # 这里只用于文档提示；真正读取列名写死了
    required_quote_columns = ("ask_price", "bid_price", "participant_timestamp", "symbol")

    @staticmethod
    def _rth_mask(ts_ns: pd.Series, tz_name: str, start: dt.time, end: dt.time) -> pd.Series:
        """
        ts_ns: pandas Series[Int64 or int]，UTC 纳秒时间戳
        返回是否在 [start, end) ET 的布尔掩码（考虑夏令时）。
        """
        ts = pd.to_datetime(ts_ns.astype("Int64"), unit="ns", utc=True)
        et = ts.dt.tz_convert(tz_name)
        h, m = et.dt.hour, et.dt.minute
        ge_start = (h > start.hour) | ((h == start.hour) & (m >= start.minute))
        lt_end   = (h < end.hour)   | ((h == end.hour)   & (m <  end.minute))
        return ge_start & lt_end

    def process_date(self, ctx: FeatureContext, date: dt.date):
        # 1) PV 样本
        sample = self.load_pv(ctx, date, columns=["symbol"])
        if sample is None:
            return None
        if sample.empty:
            return pd.DataFrame(columns=["symbol", "value"])

        # 2) 当日 onefile
        root = Path(getattr(ctx, "quote_root", self.default_quote_root))
        day_path = root / f"{date.strftime('%Y%m%d')}.parquet"
        if not day_path.exists():
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        # 3) 单次流式扫描并聚合
        tz_name = getattr(ctx, "tz", "America/New_York")
        pf = pq.ParquetFile(str(day_path))
        cols = ["symbol", "ask_price", "bid_price", "participant_timestamp"]

        sum_by: Dict[str, float] = defaultdict(float)
        cnt_by: Dict[str, int] = defaultdict(int)

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            # RTH 过滤
            rth = self._rth_mask(df["participant_timestamp"], tz_name, self.RTH_START, self.RTH_END)

            # 价格有效性与分母>0
            a = pd.to_numeric(df["ask_price"], errors="coerce")
            b = pd.to_numeric(df["bid_price"], errors="coerce")
            denom = a + b
            valid = (
                rth &
                a.replace([np.inf, -np.inf], np.nan).notna() &
                b.replace([np.inf, -np.inf], np.nan).notna() &
                denom.replace([np.inf, -np.inf], np.nan).notna() &
                (denom > 0.0)
            )

            if not bool(valid.any()):
                continue

            # 相对价差；交叉报价截断为0（避免负值）
            s = (2.0 * (a - b) / denom)[valid]

            syms = df.loc[valid, "symbol"].astype(str).values
            tmp = pd.DataFrame({"symbol": syms, "s": s})
            grp = tmp.groupby("symbol", observed=True)["s"].agg(sum="sum", count="count")

            # 累加到全局
            for k, row in grp.iterrows():
                sum_by[k] += float(row["sum"])
                cnt_by[k] += int(row["count"])

        # 4) 求均值并回填到样本顺序
        mean_by = {k: (sum_by[k] / cnt) for k, cnt in cnt_by.items() if cnt > 0}

        out = sample[["symbol"]].copy()
        out["value"] = [mean_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteSpreadOnefileFeature()
