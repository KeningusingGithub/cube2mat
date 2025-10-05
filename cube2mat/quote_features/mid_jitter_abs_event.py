# quote_features/mid_jitter_abs_event.py
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


class QuoteMidJitterAbsEventOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','ask_price','bid_price','participant_timestamp']，单次流式扫描，
    计算 RTH 内事件级中价对数收益的绝对值均值（“抖动”）：
        r_i = log(mid_i / mid_{i-1}),  jitter = E[|r_i|]（事件均值）。
    仅统计与 RTH 有正交叠时长的相邻事件；mid>0 且有限。
    输出：['symbol','value']。
    """
    name = "quote_mid_jitter_abs_event_all"
    description = "RTH event-mean of |log(mid_t / mid_{t-1})| per symbol (onefile, single pass)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = ("ask_price", "bid_price", "participant_timestamp", "symbol")

    @staticmethod
    def _rth_bounds_utc_ns(date: dt.date, tz_name: str, start: dt.time, end: dt.time):
        start_local = pd.Timestamp(dt.datetime.combine(date, start)).tz_localize(tz_name)
        end_local = pd.Timestamp(dt.datetime.combine(date, end)).tz_localize(tz_name)
        return int(start_local.tz_convert("UTC").value), int(end_local.tz_convert("UTC").value)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        sample = self.load_pv(ctx, date, columns=["symbol"])
        if sample is None:
            return None
        if sample.empty:
            return pd.DataFrame(columns=["symbol", "value"])

        root = Path(getattr(ctx, "quote_root", self.default_quote_root))
        day_path = root / f"{date.strftime('%Y%m%d')}.parquet"
        if not day_path.exists():
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        tz_name = getattr(ctx, "tz", "America/New_York")
        rth_start_ns, rth_end_ns = self._rth_bounds_utc_ns(date, tz_name, self.RTH_START, self.RTH_END)

        pf = pq.ParquetFile(str(day_path))
        cols = ["symbol", "ask_price", "bid_price", "participant_timestamp"]

        sum_by: Dict[str, float] = defaultdict(float)
        cnt_by: Dict[str, int] = defaultdict(int)

        last_ts_by: Dict[str, int] = {}
        last_mid_by: Dict[str, float] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            a = pd.to_numeric(df["ask_price"], errors="coerce")
            b = pd.to_numeric(df["bid_price"], errors="coerce")
            ts = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            mid = (a + b) / 2.0

            valid_now = (
                a.replace([np.inf, -np.inf], np.nan).notna()
                & b.replace([np.inf, -np.inf], np.nan).notna()
                & ts.replace([np.inf, -np.inf], np.nan).notna()
                & mid.replace([np.inf, -np.inf], np.nan).notna()
                & (mid > 0.0)
            )
            if not bool(valid_now.any()):
                continue

            sub = pd.DataFrame(
                {
                    "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                    "ts": ts.loc[valid_now].astype("Int64").values.astype(np.int64),
                    "mid": mid.loc[valid_now].values.astype(np.float64),
                }
            )

            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)

            p_ts = sub.groupby("symbol", sort=False)["ts"].shift(1)
            p_mid = sub.groupby("symbol", sort=False)["mid"].shift(1)

            first_mask = p_ts.isna()
            if bool(first_mask.any()):
                syms_first = sub.loc[first_mask, "symbol"].values
                p_ts.loc[first_mask] = np.array([last_ts_by.get(s) for s in syms_first], dtype="float64")
                p_mid.loc[first_mask] = np.array([last_mid_by.get(s) for s in syms_first], dtype="float64")

            t0 = p_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left = np.maximum(t0, float(rth_start_ns))
            right = np.minimum(t1, float(rth_end_ns))
            dt_ns = np.clip(right - left, 0.0, None)

            valid_pair = (
                (dt_ns > 0.0)
                & np.isfinite(p_mid.values)
                & np.isfinite(sub["mid"].values)
                & (p_mid.values > 0.0)
                & (sub["mid"].values > 0.0)
            )

            if bool(valid_pair.any()):
                r = np.log(sub["mid"].values[valid_pair] / p_mid.values[valid_pair])
                abs_r = np.abs(r)

                syms = sub["symbol"].values[valid_pair]
                tmp = (
                    pd.DataFrame({"symbol": syms, "x": abs_r})
                    .groupby("symbol", observed=True)["x"]
                    .agg(sum="sum", count="count")
                )
                for k, row in tmp.iterrows():
                    sum_by[k] += float(row["sum"])
                    cnt_by[k] += int(row["count"])

            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r0 in tail.iterrows():
                last_ts_by[r0["symbol"]] = int(r0["ts"])
                last_mid_by[r0["symbol"]] = float(r0["mid"])

        mean_by = {k: (sum_by[k] / cnt) for k, cnt in cnt_by.items() if cnt > 0}

        out = sample[["symbol"]].copy()
        out["value"] = [mean_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteMidJitterAbsEventOnefileFeature()
