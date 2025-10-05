# quote_features/tape_mid_change_zscore.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteTapeMidChangeZscoreOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','ask_price','bid_price','participant_timestamp','tape']，单次流式扫描，
    计算每个 symbol 的“中价变更强度”（次/分钟），并在各自 Tape 分组内做 z-score 标准化：
        z = (intensity - mean_tape) / std_tape
    若某 Tape 组样本过少或 std=0，则该组输出 NA。
    """

    name = "quote_tape_mid_change_z_all"
    description = "Mid-change intensity z-scored within Tape groups (A/B/C) per symbol"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = (
        "ask_price",
        "bid_price",
        "participant_timestamp",
        "tape",
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
        cols = ["symbol", "ask_price", "bid_price", "participant_timestamp", "tape"]

        time_sec_by: Dict[str, float] = {}
        chg_cnt_by: Dict[str, int] = {}
        tape_by: Dict[str, int] = {}

        last_ts_by: Dict[str, int] = {}
        last_mid_by: Dict[str, float] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            a = pd.to_numeric(df["ask_price"], errors="coerce")
            b = pd.to_numeric(df["bid_price"], errors="coerce")
            ts = pd.to_numeric(df["participant_timestamp"], errors="coerce")
            tp = pd.to_numeric(df["tape"], errors="coerce")

            mid = (a + b) / 2.0
            valid_now = (
                a.replace([np.inf, -np.inf], np.nan).notna()
                & b.replace([np.inf, -np.inf], np.nan).notna()
                & mid.replace([np.inf, -np.inf], np.nan).notna()
                & (mid > 0.0)
                & ts.replace([np.inf, -np.inf], np.nan).notna()
            )
            if not bool(valid_now.any()):
                continue

            sub = pd.DataFrame(
                {
                    "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                    "ts": ts.loc[valid_now].astype("Int64").values.astype(np.int64),
                    "mid": mid.loc[valid_now].values.astype(np.float64),
                    "tape": tp.loc[valid_now].values.astype(np.float64),
                }
            )
            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)

            for sym, g in sub.groupby("symbol", sort=False):
                if sym not in tape_by:
                    arr = g["tape"].values
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        tape_by[sym] = int(arr[0])

            p_ts = sub.groupby("symbol", sort=False)["ts"].shift(1)
            p_mid = sub.groupby("symbol", sort=False)["mid"].shift(1)

            first_mask = p_ts.isna()
            if bool(first_mask.any()):
                syms = sub.loc[first_mask, "symbol"].values
                p_ts.loc[first_mask] = np.array([last_ts_by.get(s) for s in syms], dtype="float64")
                p_mid.loc[first_mask] = np.array([last_mid_by.get(s) for s in syms], dtype="float64")

            t0 = p_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left = np.maximum(t0, float(rth_start_ns))
            right = np.minimum(t1, float(rth_end_ns))
            dt_ns = np.clip(right - left, 0.0, None)
            dt_sec = dt_ns / 1e9

            valid_pair = (
                (dt_sec > 0.0)
                & np.isfinite(p_mid.values)
                & np.isfinite(sub["mid"].values)
            )
            if bool(valid_pair.any()):
                changed = (~np.isclose(sub["mid"].values, p_mid.values)) & valid_pair
                tmp = (
                    pd.DataFrame(
                        {
                            "symbol": sub["symbol"].values,
                            "t": np.where(valid_pair, dt_sec, 0.0),
                            "c": np.where(changed, 1, 0),
                        }
                    )
                    .groupby("symbol", observed=True)
                    .sum()
                )
                for sym, row in tmp.iterrows():
                    time_sec_by[sym] = time_sec_by.get(sym, 0.0) + float(row["t"])
                    chg_cnt_by[sym] = chg_cnt_by.get(sym, 0) + int(row["c"])

            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r in tail.iterrows():
                last_ts_by[r["symbol"]] = int(r["ts"])
                last_mid_by[r["symbol"]] = float(r["mid"])

        for sym, ts_last in last_ts_by.items():
            dt_sec = max(
                0.0, (float(rth_end_ns) - max(float(ts_last), float(rth_start_ns))) / 1e9
            )
            if dt_sec > 0:
                time_sec_by[sym] = time_sec_by.get(sym, 0.0) + dt_sec

        intensity_by: Dict[str, float] = {}
        for sym, t in time_sec_by.items():
            if t > 0:
                intensity_by[sym] = chg_cnt_by.get(sym, 0) / (t / 60.0)

        tape_groups: Dict[int, list] = {}
        for sym, val in intensity_by.items():
            tp = tape_by.get(sym, None)
            if tp is None or not np.isfinite(val):
                continue
            tape_groups.setdefault(int(tp), []).append(float(val))

        tape_mean: Dict[int, float] = {}
        tape_std: Dict[int, float] = {}
        for tp, vals in tape_groups.items():
            if len(vals) >= 2:
                arr = np.array(vals, dtype=float)
                tape_mean[tp] = float(np.mean(arr))
                tape_std[tp] = float(np.std(arr, ddof=1))
            else:
                tape_mean[tp] = float(vals[0]) if len(vals) == 1 else np.nan
                tape_std[tp] = np.nan

        out = sample[["symbol"]].copy()
        zvals = []
        for sym in out["symbol"].astype(str).values:
            val = intensity_by.get(sym, np.nan)
            tp = tape_by.get(sym, None)
            mu = tape_mean.get(int(tp), np.nan) if tp is not None else np.nan
            sd = tape_std.get(int(tp), np.nan) if tp is not None else np.nan
            if np.isfinite(val) and np.isfinite(mu) and np.isfinite(sd) and sd > 0.0:
                zvals.append((val - mu) / sd)
            else:
                zvals.append(pd.NA)
        out["value"] = zvals
        return out


feature = QuoteTapeMidChangeZscoreOnefileFeature()
