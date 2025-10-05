# quote_features/price_change_share.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuotePriceChangeShareOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','bid_price','ask_price','participant_timestamp']，单次流式扫描，
    计算 RTH(09:30–16:00 ET) 的“价格变更占比”（事件占比）：
        share = # { Δbid_price ≠ 0 or Δask_price ≠ 0 } / # { 有正交叠时长的相邻事件 }
    仅统计与 RTH 有正交叠时长的相邻事件；价需有限。
    输出：['symbol','value']，∈ [0,1]。
    """
    name = "quote_price_change_share_all"
    description = "RTH event share of price-changing updates per symbol (onefile, single pass)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = ("bid_price", "ask_price", "participant_timestamp", "symbol")

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
        cols = ["symbol", "bid_price", "ask_price", "participant_timestamp"]

        tot_by: Dict[str, int] = {}
        chg_by: Dict[str, int] = {}

        last_ts_by: Dict[str, int] = {}
        last_bp_by: Dict[str, float] = {}
        last_ap_by: Dict[str, float] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            bp = pd.to_numeric(df["bid_price"], errors="coerce")
            ap = pd.to_numeric(df["ask_price"], errors="coerce")
            ts = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            valid_now = (
                bp.replace([np.inf, -np.inf], np.nan).notna()
                & ap.replace([np.inf, -np.inf], np.nan).notna()
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
                }
            )

            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)

            p_ts = sub.groupby("symbol", sort=False)["ts"].shift(1)
            p_bp = sub.groupby("symbol", sort=False)["bp"].shift(1)
            p_ap = sub.groupby("symbol", sort=False)["ap"].shift(1)

            first_mask = p_ts.isna()
            if bool(first_mask.any()):
                syms_first = sub.loc[first_mask, "symbol"].values
                p_ts.loc[first_mask] = np.array([last_ts_by.get(s, np.nan) for s in syms_first], dtype="float64")
                p_bp.loc[first_mask] = np.array([last_bp_by.get(s, np.nan) for s in syms_first], dtype="float64")
                p_ap.loc[first_mask] = np.array([last_ap_by.get(s, np.nan) for s in syms_first], dtype="float64")

            t0 = p_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left = np.maximum(t0, float(rth_start_ns))
            right = np.minimum(t1, float(rth_end_ns))
            dt_ns = np.clip(right - left, 0.0, None)

            prev_bp = p_bp.values.astype(np.float64)
            prev_ap = p_ap.values.astype(np.float64)
            cur_bp = sub["bp"].values.astype(np.float64)
            cur_ap = sub["ap"].values.astype(np.float64)

            valid_pair = (
                (dt_ns > 0.0)
                & np.isfinite(prev_bp)
                & np.isfinite(prev_ap)
                & np.isfinite(cur_bp)
                & np.isfinite(cur_ap)
            )

            if bool(valid_pair.any()):
                price_changed = (~np.isclose(cur_bp, prev_bp, rtol=price_rtol, atol=price_atol)) | (
                    ~np.isclose(cur_ap, prev_ap, rtol=price_rtol, atol=price_atol)
                )
                tmp = (
                    pd.DataFrame(
                        {
                            "symbol": sub["symbol"].values[valid_pair],
                            "tot": np.ones(int(valid_pair.sum()), dtype=np.int64),
                            "chg": price_changed[valid_pair].astype(int),
                        }
                    )
                    .groupby("symbol", observed=True)
                    .sum()
                )
                for sym, row in tmp.iterrows():
                    tot_by[sym] = tot_by.get(sym, 0) + int(row["tot"])
                    chg_by[sym] = chg_by.get(sym, 0) + int(row["chg"])

            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r0 in tail.iterrows():
                last_ts_by[r0["symbol"]] = int(r0["ts"])
                last_bp_by[r0["symbol"]] = float(r0["bp"])
                last_ap_by[r0["symbol"]] = float(r0["ap"])

        share_by: Dict[str, float] = {}
        for sym, count in tot_by.items():
            if count > 0:
                share_by[sym] = chg_by.get(sym, 0) / count

        out = sample[["symbol"]].copy()
        out["value"] = [share_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuotePriceChangeShareOnefileFeature()
