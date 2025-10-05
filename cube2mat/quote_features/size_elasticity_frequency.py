# quote_features/size_elasticity_frequency.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteSizeElasticityFrequencyOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','bid_price','ask_price','bid_size','ask_size','participant_timestamp']，单次流式扫描，
    在 RTH 内，统计“对应侧最佳价不变”的相邻事件上，尺寸发生变更(Δsize≠0) 的次数，并除以“该侧价格稳定时的总时长（分钟）”：
        rate_side = # {Δsize ≠ 0 & Δprice = 0} / 稳定时长(分钟)
    最终输出两侧(rate_bid, rate_ask)的平均（仅对有稳定时长的侧取平均）。
    输出：['symbol','value']，单位：次/分钟。
    """
    name = "quote_size_elasticity_frequency_all"
    description = "RTH frequency of size changes per minute when price is unchanged (both sides averaged)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = (
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
        "participant_timestamp",
        "symbol",
    )

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

        price_rtol = float(getattr(ctx, "price_rtol", 1e-05))
        price_atol = float(getattr(ctx, "price_atol", 1e-08))
        size_rtol = float(getattr(ctx, "size_rtol", 1e-05))
        size_atol = float(getattr(ctx, "size_atol", 1e-08))

        pf = pq.ParquetFile(str(day_path))
        cols = [
            "symbol",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
            "participant_timestamp",
        ]

        time_bid_by: Dict[str, float] = {}
        time_ask_by: Dict[str, float] = {}
        cnt_bid_by: Dict[str, int] = {}
        cnt_ask_by: Dict[str, int] = {}

        last_ts_by: Dict[str, int] = {}
        last_bp_by: Dict[str, float] = {}
        last_ap_by: Dict[str, float] = {}
        last_bsz_by: Dict[str, float] = {}
        last_asz_by: Dict[str, float] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            bp = pd.to_numeric(df["bid_price"], errors="coerce")
            ap = pd.to_numeric(df["ask_price"], errors="coerce")
            bsz = pd.to_numeric(df["bid_size"], errors="coerce")
            asz = pd.to_numeric(df["ask_size"], errors="coerce")
            ts = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            valid_now = (
                bp.replace([np.inf, -np.inf], np.nan).notna()
                & ap.replace([np.inf, -np.inf], np.nan).notna()
                & bsz.replace([np.inf, -np.inf], np.nan).notna()
                & asz.replace([np.inf, -np.inf], np.nan).notna()
                & ts.replace([np.inf, -np.inf], np.nan).notna()
            )
            if not bool(valid_now.any()):
                continue

            sub = pd.DataFrame(
                {
                    "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                    "ts": ts.loc[valid_now].astype("Int64").values.astype(np.int64),
                    "bp": bp.loc[valid_now].values.astype(np.float64),
                    "ap": ap.loc[valid_now].values.astype(np.float64),
                    "bsz": bsz.loc[valid_now].values.astype(np.float64),
                    "asz": asz.loc[valid_now].values.astype(np.float64),
                }
            )

            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)

            p_ts = sub.groupby("symbol", sort=False)["ts"].shift(1)
            p_bp = sub.groupby("symbol", sort=False)["bp"].shift(1)
            p_ap = sub.groupby("symbol", sort=False)["ap"].shift(1)
            p_bsz = sub.groupby("symbol", sort=False)["bsz"].shift(1)
            p_asz = sub.groupby("symbol", sort=False)["asz"].shift(1)

            first_mask = p_ts.isna()
            if bool(first_mask.any()):
                syms_first = sub.loc[first_mask, "symbol"].values
                p_ts.loc[first_mask] = np.array([last_ts_by.get(s, np.nan) for s in syms_first], dtype="float64")
                p_bp.loc[first_mask] = np.array([last_bp_by.get(s, np.nan) for s in syms_first], dtype="float64")
                p_ap.loc[first_mask] = np.array([last_ap_by.get(s, np.nan) for s in syms_first], dtype="float64")
                p_bsz.loc[first_mask] = np.array([last_bsz_by.get(s, np.nan) for s in syms_first], dtype="float64")
                p_asz.loc[first_mask] = np.array([last_asz_by.get(s, np.nan) for s in syms_first], dtype="float64")

            t0 = p_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left = np.maximum(t0, float(rth_start_ns))
            right = np.minimum(t1, float(rth_end_ns))
            dt_ns = np.clip(right - left, 0.0, None)
            dt_sec = dt_ns / 1e9

            prev_bp = p_bp.values.astype(np.float64)
            prev_ap = p_ap.values.astype(np.float64)
            prev_bsz = p_bsz.values.astype(np.float64)
            prev_asz = p_asz.values.astype(np.float64)
            cur_bp = sub["bp"].values.astype(np.float64)
            cur_ap = sub["ap"].values.astype(np.float64)
            cur_bsz = sub["bsz"].values.astype(np.float64)
            cur_asz = sub["asz"].values.astype(np.float64)

            valid_pair = (
                (dt_sec > 0.0)
                & np.isfinite(prev_bp)
                & np.isfinite(prev_ap)
                & np.isfinite(prev_bsz)
                & np.isfinite(prev_asz)
                & np.isfinite(cur_bp)
                & np.isfinite(cur_ap)
                & np.isfinite(cur_bsz)
                & np.isfinite(cur_asz)
            )

            if bool(valid_pair.any()):
                bid_stable = valid_pair & np.isclose(cur_bp, prev_bp, rtol=price_rtol, atol=price_atol)
                ask_stable = valid_pair & np.isclose(cur_ap, prev_ap, rtol=price_rtol, atol=price_atol)

                if bool(bid_stable.any()):
                    tb = dt_sec[bid_stable]
                    changed = ~np.isclose(cur_bsz, prev_bsz, rtol=size_rtol, atol=size_atol)
                    cb = changed[bid_stable].astype(int)
                    tmp = (
                        pd.DataFrame(
                            {
                                "symbol": sub["symbol"].values[bid_stable],
                                "time": tb,
                                "cnt": cb,
                            }
                        )
                        .groupby("symbol", observed=True)
                        .sum()
                    )
                    for sym, row in tmp.iterrows():
                        time_bid_by[sym] = time_bid_by.get(sym, 0.0) + float(row["time"])
                        cnt_bid_by[sym] = cnt_bid_by.get(sym, 0) + int(row["cnt"])

                if bool(ask_stable.any()):
                    ta = dt_sec[ask_stable]
                    changed = ~np.isclose(cur_asz, prev_asz, rtol=size_rtol, atol=size_atol)
                    ca = changed[ask_stable].astype(int)
                    tmp = (
                        pd.DataFrame(
                            {
                                "symbol": sub["symbol"].values[ask_stable],
                                "time": ta,
                                "cnt": ca,
                            }
                        )
                        .groupby("symbol", observed=True)
                        .sum()
                    )
                    for sym, row in tmp.iterrows():
                        time_ask_by[sym] = time_ask_by.get(sym, 0.0) + float(row["time"])
                        cnt_ask_by[sym] = cnt_ask_by.get(sym, 0) + int(row["cnt"])

            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r0 in tail.iterrows():
                last_ts_by[r0["symbol"]] = int(r0["ts"])
                last_bp_by[r0["symbol"]] = float(r0["bp"])
                last_ap_by[r0["symbol"]] = float(r0["ap"])
                last_bsz_by[r0["symbol"]] = float(r0["bsz"])
                last_asz_by[r0["symbol"]] = float(r0["asz"])

        for sym, ts_last in last_ts_by.items():
            left = max(float(ts_last), float(rth_start_ns))
            right = float(rth_end_ns)
            dt_sec = max(0.0, (right - left) / 1e9)
            if dt_sec > 0:
                time_bid_by[sym] = time_bid_by.get(sym, 0.0) + dt_sec
                time_ask_by[sym] = time_ask_by.get(sym, 0.0) + dt_sec

        rate_by: Dict[str, float] = {}
        keys = set(time_bid_by) | set(time_ask_by)
        for sym in keys:
            tb = time_bid_by.get(sym, 0.0)
            ta = time_ask_by.get(sym, 0.0)
            rb = cnt_bid_by.get(sym, 0) / (tb / 60.0) if tb > 0 else np.nan
            ra = cnt_ask_by.get(sym, 0) / (ta / 60.0) if ta > 0 else np.nan
            if np.isfinite(rb) and np.isfinite(ra):
                rate_by[sym] = 0.5 * (rb + ra)
            elif np.isfinite(rb):
                rate_by[sym] = rb
            elif np.isfinite(ra):
                rate_by[sym] = ra

        out = sample[["symbol"]].copy()
        out["value"] = [rate_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteSizeElasticityFrequencyOnefileFeature()
