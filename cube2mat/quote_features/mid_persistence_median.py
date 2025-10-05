# quote_features/mid_persistence_median.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteMidPersistenceMedianOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','ask_price','bid_price','participant_timestamp']，单次流式扫描，
    估计 RTH 内“中价保持不变的稳定期时长”的中位数（单位：秒），使用每 symbol 的固定宽直方图近似：
      - 在区间 [t_{i-1}, t_i) 内，中价视为持有 mid_{i-1}；
      - 若在事件 i 时发生 mid 变更，则前一段稳定期结束，其累计时长计入直方图；
      - 尾段 [t_last, 16:00) 也计入最后一个稳定期；
    直方图上界默认 3600s（1小时），超出部分截断到上限；可通过 ctx.persist_max_sec 覆盖。
    输出：['symbol','value']，为各自直方图估计的 p50（中位数）。
    """
    name = "quote_mid_persistence_median_all"
    description = "RTH median duration (sec) of constant mid-price, per symbol (histogram approximation, onefile)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    DEFAULT_MAX_SEC = 3600.0
    N_BINS = 512

    required_pv_columns = ("symbol",)
    required_quote_columns = ("ask_price", "bid_price", "participant_timestamp", "symbol")

    @staticmethod
    def _rth_bounds_utc_ns(date: dt.date, tz_name: str, start: dt.time, end: dt.time) -> Tuple[int, int]:
        start_local = pd.Timestamp(dt.datetime.combine(date, start)).tz_localize(tz_name)
        end_local = pd.Timestamp(dt.datetime.combine(date, end)).tz_localize(tz_name)
        return int(start_local.tz_convert("UTC").value), int(end_local.tz_convert("UTC").value)

    @staticmethod
    def _bin_index_sec(x_sec: np.ndarray, vmax: float, n_bins: int) -> np.ndarray:
        if vmax <= 0:
            return np.zeros_like(x_sec, dtype=np.int64)
        x = np.clip(x_sec, 0.0, vmax)
        idx = np.floor((x / vmax) * n_bins).astype(np.int64)
        idx = np.minimum(idx, n_bins - 1)
        return idx

    @staticmethod
    def _p50_from_hist(counts: np.ndarray, vmax: float) -> float:
        total = counts.sum()
        if total <= 0:
            return np.nan
        k = int(np.ceil(0.5 * total))
        csum = counts.cumsum()
        b = int(np.searchsorted(csum, k, side="left"))
        return vmax * (b + 1) / len(counts)

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
        vmax = float(getattr(ctx, "persist_max_sec", self.DEFAULT_MAX_SEC))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = self.DEFAULT_MAX_SEC
        rth_start_ns, rth_end_ns = self._rth_bounds_utc_ns(date, tz_name, self.RTH_START, self.RTH_END)

        pf = pq.ParquetFile(str(day_path))
        cols = ["symbol", "ask_price", "bid_price", "participant_timestamp"]

        hist_by: Dict[str, np.ndarray] = {}
        run_acc_sec_by: Dict[str, float] = {}
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

            for sym, g in sub.groupby("symbol", sort=False):
                arr_ts = g["ts"].values.astype(np.int64)
                arr_mid = g["mid"].values.astype(np.float64)

                if sym not in hist_by:
                    hist_by[sym] = np.zeros(self.N_BINS, dtype=np.int64)

                prev_ts = float(last_ts_by.get(sym, np.nan))
                prev_mid = float(last_mid_by.get(sym, np.nan))
                carry = float(run_acc_sec_by.get(sym, 0.0))

                for k in range(len(arr_ts)):
                    cur_ts = float(arr_ts[k])
                    cur_mid = float(arr_mid[k])

                    if np.isfinite(prev_ts) and np.isfinite(prev_mid):
                        left = max(prev_ts, float(rth_start_ns))
                        right = min(cur_ts, float(rth_end_ns))
                        dt_sec = max(0.0, (right - left) / 1e9)
                    else:
                        dt_sec = 0.0

                    carry += dt_sec

                    changed = (
                        np.isfinite(prev_mid)
                        and np.isfinite(cur_mid)
                        and (not np.isclose(cur_mid, prev_mid))
                    )

                    if changed and carry > 0.0:
                        idx = self._bin_index_sec(np.array([carry]), vmax, self.N_BINS)[0]
                        hist_by[sym][idx] += 1
                        carry = 0.0

                    prev_ts = cur_ts
                    prev_mid = cur_mid

                last_ts_by[sym] = int(prev_ts) if np.isfinite(prev_ts) else last_ts_by.get(sym, 0)
                last_mid_by[sym] = float(prev_mid) if np.isfinite(prev_mid) else last_mid_by.get(sym, np.nan)
                run_acc_sec_by[sym] = float(carry)

        for sym, ts_last in last_ts_by.items():
            prev_ts = float(ts_last)
            prev_mid = float(last_mid_by.get(sym, np.nan))
            carry = float(run_acc_sec_by.get(sym, 0.0))

            if np.isfinite(prev_ts) and np.isfinite(prev_mid):
                left = max(prev_ts, float(rth_start_ns))
                right = float(rth_end_ns)
                dt_sec = max(0.0, (right - left) / 1e9)
                carry += dt_sec

            if carry > 0.0:
                if sym not in hist_by:
                    hist_by[sym] = np.zeros(self.N_BINS, dtype=np.int64)
                idx = self._bin_index_sec(np.array([carry]), vmax, self.N_BINS)[0]
                hist_by[sym][idx] += 1
                run_acc_sec_by[sym] = 0.0

        med_by: Dict[str, float] = {}
        for sym, counts in hist_by.items():
            val = self._p50_from_hist(counts, vmax)
            if np.isfinite(val):
                med_by[sym] = float(val)

        out = sample[["symbol"]].copy()
        out["value"] = [med_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteMidPersistenceMedianOnefileFeature()
