# quote_features/size_elasticity_magnitude.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteSizeElasticityMagnitudeOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','bid_price','ask_price','bid_size','ask_size','participant_timestamp']，单次流式扫描，
    在 RTH 内、且“对应侧最佳价不变”的相邻事件上，计算 |Δsize| 的事件均值：
        对 bid 侧：当 Δbid_price=0 时计入 |Δbid_size|
        对 ask 侧：当 Δask_price=0 时计入 |Δask_size|
    最终输出两侧合并的事件均值。
    输出：['symbol','value']，单位：手（lots）。
    """
    name = "quote_size_elasticity_magnitude_all"
    description = "RTH event-mean of |Δsize| conditional on no price change (both sides combined)"
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

        pf = pq.ParquetFile(str(day_path))
        cols = [
            "symbol",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
            "participant_timestamp",
        ]

        sum_by: Dict[str, float] = {}
        cnt_by: Dict[str, int] = {}

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

            prev_bp = p_bp.values.astype(np.float64)
            prev_ap = p_ap.values.astype(np.float64)
            prev_bsz = p_bsz.values.astype(np.float64)
            prev_asz = p_asz.values.astype(np.float64)
            cur_bp = sub["bp"].values.astype(np.float64)
            cur_ap = sub["ap"].values.astype(np.float64)
            cur_bsz = sub["bsz"].values.astype(np.float64)
            cur_asz = sub["asz"].values.astype(np.float64)

            valid_pair = (
                (dt_ns > 0.0)
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
                    vals = np.abs(cur_bsz[bid_stable] - prev_bsz[bid_stable])
                    if vals.size:
                        tmp = (
                            pd.DataFrame(
                                {
                                    "symbol": sub["symbol"].values[bid_stable],
                                    "val": vals,
                                }
                            )
                            .groupby("symbol", observed=True)["val"]
                            .agg(sum="sum", count="count")
                        )
                        for sym, row in tmp.iterrows():
                            sum_by[sym] = sum_by.get(sym, 0.0) + float(row["sum"])
                            cnt_by[sym] = cnt_by.get(sym, 0) + int(row["count"])

                if bool(ask_stable.any()):
                    vals = np.abs(cur_asz[ask_stable] - prev_asz[ask_stable])
                    if vals.size:
                        tmp = (
                            pd.DataFrame(
                                {
                                    "symbol": sub["symbol"].values[ask_stable],
                                    "val": vals,
                                }
                            )
                            .groupby("symbol", observed=True)["val"]
                            .agg(sum="sum", count="count")
                        )
                        for sym, row in tmp.iterrows():
                            sum_by[sym] = sum_by.get(sym, 0.0) + float(row["sum"])
                            cnt_by[sym] = cnt_by.get(sym, 0) + int(row["count"])

            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r0 in tail.iterrows():
                last_ts_by[r0["symbol"]] = int(r0["ts"])
                last_bp_by[r0["symbol"]] = float(r0["bp"])
                last_ap_by[r0["symbol"]] = float(r0["ap"])
                last_bsz_by[r0["symbol"]] = float(r0["bsz"])
                last_asz_by[r0["symbol"]] = float(r0["asz"])

        mean_by = {sym: (sum_by[sym] / cnt) for sym, cnt in cnt_by.items() if cnt > 0}

        out = sample[["symbol"]].copy()
        out["value"] = [mean_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteSizeElasticityMagnitudeOnefileFeature()
