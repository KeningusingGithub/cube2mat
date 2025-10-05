# quote_features/queue_depletion_rate.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteQueueDepletionRateOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','bid_price','ask_price','bid_size','ask_size','participant_timestamp']，单次流式扫描，
    计算 RTH 内“最佳价不变时”的队列消耗速率（单位：手/秒）：
        对相邻事件 i-1 -> i，若 bid_price 不变：
            计入 bid 侧的时长 dt = overlap([t_{i-1}, t_i), RTH)
            计入消耗量 vol = max(0, bid_size_{i-1} - bid_size_i)
        ask 侧同理。
    聚合：rate_side = (∑vol_side) / (∑dt_side)；输出 value = 平均(rate_bid, rate_ask)（仅对有时长的侧取平均）。
    为避免 DT=0 或缺失值引发问题，严格筛选有效样本；尾段默认无额外消耗，但将最后状态的时长补到 16:00（只计时长）。
    """
    name = "quote_queue_depletion_rate_all"
    description = "RTH queue depletion rate when best price unchanged (lots per second) per symbol (onefile, single pass)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END   = dt.time(16,  0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = ("bid_price", "ask_price", "bid_size", "ask_size", "participant_timestamp", "symbol")

    @staticmethod
    def _rth_bounds_utc_ns(date: dt.date, tz_name: str, start: dt.time, end: dt.time) -> Tuple[int, int]:
        start_local = pd.Timestamp(dt.datetime.combine(date, start)).tz_localize(tz_name)
        end_local   = pd.Timestamp(dt.datetime.combine(date, end)).tz_localize(tz_name)
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
        cols = ["symbol", "bid_price", "ask_price", "bid_size", "ask_size", "participant_timestamp"]

        vol_bid_by: Dict[str, float] = {}
        time_bid_by: Dict[str, float] = {}
        vol_ask_by: Dict[str, float] = {}
        time_ask_by: Dict[str, float] = {}

        last_ts_by: Dict[str, int] = {}
        last_bp_by: Dict[str, float] = {}
        last_ap_by: Dict[str, float] = {}
        last_bsz_by: Dict[str, float] = {}
        last_asz_by: Dict[str, float] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            bp  = pd.to_numeric(df["bid_price"], errors="coerce")
            ap  = pd.to_numeric(df["ask_price"], errors="coerce")
            bsz = pd.to_numeric(df["bid_size"], errors="coerce")
            asz = pd.to_numeric(df["ask_size"], errors="coerce")
            ts  = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            valid_now = (
                bp.replace([np.inf, -np.inf], np.nan).notna() &
                ap.replace([np.inf, -np.inf], np.nan).notna() &
                bsz.replace([np.inf, -np.inf], np.nan).notna() &
                asz.replace([np.inf, -np.inf], np.nan).notna() &
                ts.replace([np.inf, -np.inf], np.nan).notna()
            )
            if not bool(valid_now.any()):
                continue

            sub = pd.DataFrame({
                "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                "ts":  ts.loc[valid_now].astype("Int64").values.astype(np.int64),
                "bp":  bp.loc[valid_now].values.astype(np.float64),
                "ap":  ap.loc[valid_now].values.astype(np.float64),
                "bsz": bsz.loc[valid_now].values.astype(np.float64),
                "asz": asz.loc[valid_now].values.astype(np.float64),
            })

            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)
            p_ts  = sub.groupby("symbol", sort=False)["ts"].shift(1)
            p_bp  = sub.groupby("symbol", sort=False)["bp"].shift(1)
            p_ap  = sub.groupby("symbol", sort=False)["ap"].shift(1)
            p_bsz = sub.groupby("symbol", sort=False)["bsz"].shift(1)
            p_asz = sub.groupby("symbol", sort=False)["asz"].shift(1)

            first_mask = p_ts.isna()
            if bool(first_mask.any()):
                syms_first = sub.loc[first_mask, "symbol"].values
                p_ts.loc[first_mask]  = np.array([last_ts_by.get(s)  for s in syms_first], dtype="float64")
                p_bp.loc[first_mask]  = np.array([last_bp_by.get(s)  for s in syms_first], dtype="float64")
                p_ap.loc[first_mask]  = np.array([last_ap_by.get(s)  for s in syms_first], dtype="float64")
                p_bsz.loc[first_mask] = np.array([last_bsz_by.get(s) for s in syms_first], dtype="float64")
                p_asz.loc[first_mask] = np.array([last_asz_by.get(s) for s in syms_first], dtype="float64")

            t0 = p_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left  = np.maximum(t0, float(rth_start_ns))
            right = np.minimum(t1, float(rth_end_ns))
            dt_ns = np.clip(right - left, 0.0, None)
            dt_sec = dt_ns / 1e9

            bid_stable = (
                np.isfinite(p_bp.values) & np.isfinite(sub["bp"].values) &
                (sub["bp"].values == p_bp.values) & (dt_sec > 0.0)
            )
            ask_stable = (
                np.isfinite(p_ap.values) & np.isfinite(sub["ap"].values) &
                (sub["ap"].values == p_ap.values) & (dt_sec > 0.0)
            )

            bid_valid = bid_stable & np.isfinite(p_bsz.values) & np.isfinite(sub["bsz"].values)
            ask_valid = ask_stable & np.isfinite(p_asz.values) & np.isfinite(sub["asz"].values)

            time_bid = np.where(bid_valid, dt_sec, 0.0)
            time_ask = np.where(ask_valid, dt_sec, 0.0)
            depl_bid = np.where(bid_valid, np.maximum(0.0, p_bsz.values - sub["bsz"].values), 0.0)
            depl_ask = np.where(ask_valid, np.maximum(0.0, p_asz.values - sub["asz"].values), 0.0)

            agg = pd.DataFrame({
                "symbol": sub["symbol"].values,
                "tb": time_bid,
                "vb": depl_bid,
                "ta": time_ask,
                "va": depl_ask,
            }).groupby("symbol", observed=True).sum()

            for sym, row in agg.iterrows():
                time_bid_by[sym] = time_bid_by.get(sym, 0.0) + float(row["tb"])
                vol_bid_by[sym]  = vol_bid_by.get(sym, 0.0)  + float(row["vb"])
                time_ask_by[sym] = time_ask_by.get(sym, 0.0) + float(row["ta"])
                vol_ask_by[sym]  = vol_ask_by.get(sym, 0.0)  + float(row["va"])

            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r in tail.iterrows():
                sym = r["symbol"]
                last_ts_by[sym]  = int(r["ts"])
                last_bp_by[sym]  = float(r["bp"])
                last_ap_by[sym]  = float(r["ap"])
                last_bsz_by[sym] = float(r["bsz"])
                last_asz_by[sym] = float(r["asz"])

        for sym, ts_last in last_ts_by.items():
            left  = max(float(ts_last), float(rth_start_ns))
            right = float(rth_end_ns)
            dt_sec = max(0.0, (right - left) / 1e9)
            if dt_sec > 0:
                time_bid_by[sym] = time_bid_by.get(sym, 0.0) + dt_sec
                time_ask_by[sym] = time_ask_by.get(sym, 0.0) + dt_sec

        rate_by: Dict[str, float] = {}
        keys = set(list(time_bid_by.keys()) + list(time_ask_by.keys()))
        for sym in keys:
            rb = vol_bid_by.get(sym, 0.0) / time_bid_by.get(sym, np.nan) if time_bid_by.get(sym, 0.0) > 0 else np.nan
            ra = vol_ask_by.get(sym, 0.0) / time_ask_by.get(sym, np.nan) if time_ask_by.get(sym, 0.0) > 0 else np.nan
            if np.isfinite(rb) and np.isfinite(ra):
                rate_by[sym] = 0.5 * (rb + ra)
            elif np.isfinite(rb):
                rate_by[sym] = rb
            elif np.isfinite(ra):
                rate_by[sym] = ra

        out = sample[["symbol"]].copy()
        out["value"] = [rate_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteQueueDepletionRateOnefileFeature()
