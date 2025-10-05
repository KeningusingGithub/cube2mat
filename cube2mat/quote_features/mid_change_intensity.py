# quote_features/mid_change_intensity.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteMidChangeIntensityOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','ask_price','bid_price','participant_timestamp']，单次流式扫描，
    计算 RTH(09:30–16:00 ET) 的“中价变动强度”：中价变更次数 / 有效观测时长（分钟）。
    有效观测时长按区间 [t_{i-1}, t_i) 与 RTH 的交叠累计，尾段补到 16:00。
    仅在交叠时长 > 0 的相邻事件上计一次“变更”，且要求 mid_{i-1} 与 mid_i 有限、正。
    输出：['symbol','value']，单位：次/分钟。
    """
    name = "quote_mid_change_intensity_all"
    description = "RTH rate of mid-price changes (changes per minute) per symbol (onefile, single pass)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = ("ask_price", "bid_price", "participant_timestamp", "symbol")

    @staticmethod
    def _rth_bounds_utc_ns(date: dt.date, tz_name: str, start: dt.time, end: dt.time) -> Tuple[int, int]:
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

        time_sec_by: Dict[str, float] = {}
        change_cnt_by: Dict[str, int] = {}

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
            dt_sec = dt_ns / 1e9

            valid_pair = (dt_sec > 0.0) & np.isfinite(p_mid.values) & np.isfinite(sub["mid"].values)
            if bool(valid_pair.any()):
                changed = (~np.isclose(sub["mid"].values, p_mid.values)) & valid_pair

                tmp = pd.DataFrame(
                    {
                        "symbol": sub["symbol"].values,
                        "t": np.where(valid_pair, dt_sec, 0.0),
                        "c": np.where(changed, 1, 0),
                    }
                ).groupby("symbol", observed=True).sum()

                for sym, row in tmp.iterrows():
                    time_sec_by[sym] = time_sec_by.get(sym, 0.0) + float(row["t"])
                    change_cnt_by[sym] = change_cnt_by.get(sym, 0) + int(row["c"])

            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r in tail.iterrows():
                last_ts_by[r["symbol"]] = int(r["ts"])
                last_mid_by[r["symbol"]] = float(r["mid"])

        for sym, ts_last in last_ts_by.items():
            left = max(float(ts_last), float(rth_start_ns))
            right = float(rth_end_ns)
            dt_sec = max(0.0, (right - left) / 1e9)
            if dt_sec > 0:
                time_sec_by[sym] = time_sec_by.get(sym, 0.0) + dt_sec

        rate_by: Dict[str, float] = {}
        for sym, t in time_sec_by.items():
            if t > 0:
                rate_by[sym] = change_cnt_by.get(sym, 0) / (t / 60.0)

        out = sample[["symbol"]].copy()
        out["value"] = [rate_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteMidChangeIntensityOnefileFeature()
