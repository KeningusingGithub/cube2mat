# quote_features/venue_switch_rate.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteVenueSwitchRateOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','ask_exchange','bid_exchange','participant_timestamp']，单次流式扫描，
    计算 RTH(09:30–16:00 ET) 的“交易所切换频率”（双侧合并）：
        每当 ask_exchange 或 bid_exchange 相对上一事件发生变化，计数 +1；
        频率 = 切换次数 / 观测时长(分钟)。
    输出：['symbol','value']，单位：次/分钟；若无观测时长则 NA。
    """

    name = "quote_venue_switch_rate_all"
    description = (
        "RTH rate of venue switches (ask/bid exchange changes per minute), both sides combined"
    )
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = (
        "ask_exchange",
        "bid_exchange",
        "participant_timestamp",
        "symbol",
    )

    @staticmethod
    def _rth_bounds_utc_ns(
        date: dt.date, tz_name: str, start: dt.time, end: dt.time
    ) -> Tuple[int, int]:
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
        rth_start_ns, rth_end_ns = self._rth_bounds_utc_ns(
            date, tz_name, self.RTH_START, self.RTH_END
        )

        pf = pq.ParquetFile(str(day_path))
        cols = ["symbol", "ask_exchange", "bid_exchange", "participant_timestamp"]

        time_sec_by: Dict[str, float] = {}
        switch_cnt_by: Dict[str, int] = {}

        last_ts_by: Dict[str, int] = {}
        last_ae_by: Dict[str, float] = {}
        last_be_by: Dict[str, float] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            ae = pd.to_numeric(df["ask_exchange"], errors="coerce")
            be = pd.to_numeric(df["bid_exchange"], errors="coerce")
            ts = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            valid_now = ts.replace([np.inf, -np.inf], np.nan).notna()
            if not bool(valid_now.any()):
                continue

            sub = pd.DataFrame(
                {
                    "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                    "ts": ts.loc[valid_now].astype("Int64").values.astype(np.int64),
                    "ae": ae.loc[valid_now].values.astype(np.float64),
                    "be": be.loc[valid_now].values.astype(np.float64),
                }
            )
            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)

            p_ts = sub.groupby("symbol", sort=False)["ts"].shift(1)
            p_ae = sub.groupby("symbol", sort=False)["ae"].shift(1)
            p_be = sub.groupby("symbol", sort=False)["be"].shift(1)

            first_mask = p_ts.isna()
            if bool(first_mask.any()):
                syms = sub.loc[first_mask, "symbol"].values
                p_ts.loc[first_mask] = np.array([last_ts_by.get(s) for s in syms], dtype="float64")
                p_ae.loc[first_mask] = np.array([last_ae_by.get(s) for s in syms], dtype="float64")
                p_be.loc[first_mask] = np.array([last_be_by.get(s) for s in syms], dtype="float64")

            t0 = p_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left = np.maximum(t0, float(rth_start_ns))
            right = np.minimum(t1, float(rth_end_ns))
            dt_ns = np.clip(right - left, 0.0, None)
            dt_sec = dt_ns / 1e9

            valid_pair = dt_sec > 0.0

            if bool(valid_pair.any()):
                changed_ae = (
                    valid_pair
                    & np.isfinite(p_ae.values)
                    & np.isfinite(sub["ae"].values)
                    & (sub["ae"].values != p_ae.values)
                )
                changed_be = (
                    valid_pair
                    & np.isfinite(p_be.values)
                    & np.isfinite(sub["be"].values)
                    & (sub["be"].values != p_be.values)
                )

                tmp = (
                    pd.DataFrame(
                        {
                            "symbol": sub["symbol"].values,
                            "t": np.where(valid_pair, dt_sec, 0.0),
                            "c": np.where(changed_ae, 1, 0) + np.where(changed_be, 1, 0),
                        }
                    )
                    .groupby("symbol", observed=True)
                    .sum()
                )

                for sym, row in tmp.iterrows():
                    time_sec_by[sym] = time_sec_by.get(sym, 0.0) + float(row["t"])
                    switch_cnt_by[sym] = switch_cnt_by.get(sym, 0) + int(row["c"])

            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r in tail.iterrows():
                last_ts_by[r["symbol"]] = int(r["ts"])
                last_ae_by[r["symbol"]] = float(r["ae"])
                last_be_by[r["symbol"]] = float(r["be"])

        for sym, ts_last in last_ts_by.items():
            dt_sec = max(
                0.0, (float(rth_end_ns) - max(float(ts_last), float(rth_start_ns))) / 1e9
            )
            if dt_sec > 0:
                time_sec_by[sym] = time_sec_by.get(sym, 0.0) + dt_sec

        out = sample[["symbol"]].copy()
        vals = []
        for sym in out["symbol"].astype(str).values:
            t = time_sec_by.get(sym, 0.0)
            if t > 0:
                vals.append(switch_cnt_by.get(sym, 0) / (t / 60.0))
            else:
                vals.append(pd.NA)
        out["value"] = vals
        return out


feature = QuoteVenueSwitchRateOnefileFeature()
