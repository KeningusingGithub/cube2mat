# quote_features/same_venue_share_tw.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteSameVenueShareTWOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','ask_exchange','bid_exchange','participant_timestamp']，单次流式扫描，
    计算 RTH(09:30–16:00 ET) 的“同场所占比”（时间加权）：
      share = time{ ask_exchange == bid_exchange } / time{ 两侧场所均有效 }
    使用上一事件状态在区间 [t_{i-1}, t_i) 持有；尾段补到 16:00。
    输出：['symbol','value'] ∈ [0,1]；若分母=0 则 NA。
    """

    name = "quote_same_venue_share_tw_all"
    description = "RTH time share where ask and bid are on the same venue (TW)"
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

        same_by: Dict[str, float] = {}
        valid_by: Dict[str, float] = {}

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

            both_valid = (
                (dt_ns > 0.0)
                & np.isfinite(p_ae.values)
                & np.isfinite(p_be.values)
            )
            same = both_valid & (p_ae.values == p_be.values)

            if bool(both_valid.any()):
                tmp = (
                    pd.DataFrame(
                        {
                            "symbol": sub["symbol"].values,
                            "v": np.where(both_valid, dt_ns, 0.0),
                            "s": np.where(same, dt_ns, 0.0),
                        }
                    )
                    .groupby("symbol", observed=True)
                    .sum()
                )

                for sym, row in tmp.iterrows():
                    valid_by[sym] = valid_by.get(sym, 0.0) + float(row["v"])
                    same_by[sym] = same_by.get(sym, 0.0) + float(row["s"])

            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r in tail.iterrows():
                last_ts_by[r["symbol"]] = int(r["ts"])
                last_ae_by[r["symbol"]] = float(r["ae"])
                last_be_by[r["symbol"]] = float(r["be"])

        for sym, ts_last in last_ts_by.items():
            dt_ns = max(0.0, float(rth_end_ns) - max(float(ts_last), float(rth_start_ns)))
            if dt_ns <= 0:
                continue
            if np.isfinite(last_ae_by.get(sym, np.nan)) and np.isfinite(
                last_be_by.get(sym, np.nan)
            ):
                valid_by[sym] = valid_by.get(sym, 0.0) + dt_ns
                if int(last_ae_by[sym]) == int(last_be_by[sym]):
                    same_by[sym] = same_by.get(sym, 0.0) + dt_ns

        out = sample[["symbol"]].copy()
        vals = []
        for sym in out["symbol"].astype(str).values:
            denom = valid_by.get(sym, 0.0)
            if denom > 0:
                vals.append(same_by.get(sym, 0.0) / denom)
            else:
                vals.append(pd.NA)
        out["value"] = vals
        return out


feature = QuoteSameVenueShareTWOnefileFeature()
