# quote_features/quote_update_intensity.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteUpdateIntensityOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','sequence_number','participant_timestamp']，单次流式扫描，
    计算 RTH(09:30–16:00 ET) 的“报价事件强度”（QPS）：
        rate = (∑ Δsequence_number) / (观测时长分钟)
    其中观测时长为相邻事件区间与 RTH 的交叠时长之和；尾段补到 16:00。
    Δsequence_number<0 视为 0（异常或重置）。
    输出：['symbol','value']，单位：事件/分钟。
    """
    name = "quote_update_intensity_all"
    description = "RTH rate of quote updates using Δsequence_number per minute (onefile, single pass)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = ("sequence_number", "participant_timestamp", "symbol")

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
        cols = ["symbol", "sequence_number", "participant_timestamp"]

        ev_sum_by: Dict[str, float] = {}
        time_sec_by: Dict[str, float] = {}

        last_ts_by: Dict[str, int] = {}
        last_seq_by: Dict[str, float] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            seq = pd.to_numeric(df["sequence_number"], errors="coerce")
            ts = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            valid_now = (
                seq.replace([np.inf, -np.inf], np.nan).notna()
                & ts.replace([np.inf, -np.inf], np.nan).notna()
            )
            if not bool(valid_now.any()):
                continue

            sub = pd.DataFrame(
                {
                    "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                    "ts": ts.loc[valid_now].astype("Int64").values.astype(np.int64),
                    "seq": seq.loc[valid_now].values.astype(np.float64),
                }
            )

            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)

            p_ts = sub.groupby("symbol", sort=False)["ts"].shift(1)
            p_seq = sub.groupby("symbol", sort=False)["seq"].shift(1)

            first_mask = p_ts.isna()
            if bool(first_mask.any()):
                syms_first = sub.loc[first_mask, "symbol"].values
                p_ts.loc[first_mask] = np.array([last_ts_by.get(s, np.nan) for s in syms_first], dtype="float64")
                p_seq.loc[first_mask] = np.array([last_seq_by.get(s, np.nan) for s in syms_first], dtype="float64")

            t0 = p_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left = np.maximum(t0, float(rth_start_ns))
            right = np.minimum(t1, float(rth_end_ns))
            dt_ns = np.clip(right - left, 0.0, None)
            dt_sec = dt_ns / 1e9

            prev_seq = p_seq.values.astype(np.float64)
            cur_seq = sub["seq"].values.astype(np.float64)
            dseq = cur_seq - prev_seq
            dseq = np.where(np.isfinite(dseq), dseq, np.nan)
            dseq = np.where(dseq < 0.0, 0.0, dseq)

            valid_pair = (dt_sec > 0.0) & np.isfinite(dseq)

            if bool(valid_pair.any()):
                tmp = (
                    pd.DataFrame(
                        {
                            "symbol": sub["symbol"].values[valid_pair],
                            "ev": dseq[valid_pair],
                            "t": dt_sec[valid_pair],
                        }
                    )
                    .groupby("symbol", observed=True)
                    .sum()
                )
                for sym, row in tmp.iterrows():
                    ev_sum_by[sym] = ev_sum_by.get(sym, 0.0) + float(row["ev"])
                    time_sec_by[sym] = time_sec_by.get(sym, 0.0) + float(row["t"])

            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r0 in tail.iterrows():
                last_ts_by[r0["symbol"]] = int(r0["ts"])
                last_seq_by[r0["symbol"]] = float(r0["seq"])

        for sym, ts_last in last_ts_by.items():
            left = max(float(ts_last), float(rth_start_ns))
            right = float(rth_end_ns)
            dt_sec = max(0.0, (right - left) / 1e9)
            if dt_sec > 0:
                time_sec_by[sym] = time_sec_by.get(sym, 0.0) + dt_sec

        rate_by: Dict[str, float] = {}
        for sym, t in time_sec_by.items():
            if t > 0:
                rate_by[sym] = ev_sum_by.get(sym, 0.0) / (t / 60.0)

        out = sample[["symbol"]].copy()
        out["value"] = [rate_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out

feature = QuoteUpdateIntensityOnefileFeature()
