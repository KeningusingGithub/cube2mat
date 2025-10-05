# quote_features/locked_crossed_share_tw.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteLockedCrossedShareTWOnefileFeature(QuoteBaseFeature):
    """
    在 RTH(09:30–16:00 ET) 内，时间加权计算 “被锁/交叉”(ask<=bid) 的占比：
      share = time{ ask<=bid } / time{ 观测到的 RTH 时长 }
    以上一事件的状态在 [t_{i-1}, t_i) 持有；尾段补到 16:00。
    输出：['symbol','value']，∈[0,1]。
    """
    name = "quote_locked_crossed_share_tw_all"
    description = "RTH time share where the book is locked/crossed (ask<=bid), piecewise-constant (onefile)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END   = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = ("ask_price", "bid_price", "participant_timestamp", "symbol")

    @staticmethod
    def _rth_bounds_utc_ns(date: dt.date, tz_name: str, start: dt.time, end: dt.time) -> Tuple[int, int]:
        s = pd.Timestamp(dt.datetime.combine(date, start)).tz_localize(tz_name)
        e = pd.Timestamp(dt.datetime.combine(date, end)).tz_localize(tz_name)
        return int(s.tz_convert("UTC").value), int(e.tz_convert("UTC").value)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        sample = self.load_pv(ctx, date, columns=["symbol"])
        if sample is None: return None
        if sample.empty: return pd.DataFrame(columns=["symbol", "value"])

        root = Path(getattr(ctx, "quote_root", self.default_quote_root))
        path = root / f"{date.strftime('%Y%m%d')}.parquet"
        if not path.exists():
            out = sample[["symbol"]].copy(); out["value"] = pd.NA; return out

        tz = getattr(ctx, "tz", "America/New_York")
        T0, T1 = self._rth_bounds_utc_ns(date, tz, self.RTH_START, self.RTH_END)

        pf = pq.ParquetFile(str(path))
        cols = ["symbol", "ask_price", "bid_price", "participant_timestamp"]

        # 累加器
        lock_ns_by: Dict[str, float] = {}
        time_ns_by: Dict[str, float] = {}

        # 上一次状态
        last_ts_by: Dict[str, int] = {}
        last_is_locked_by: Dict[str, bool] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()
            a  = pd.to_numeric(df["ask_price"], errors="coerce")
            b  = pd.to_numeric(df["bid_price"], errors="coerce")
            ts = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            valid_now = a.replace([np.inf, -np.inf], np.nan).notna() & \
                        b.replace([np.inf, -np.inf], np.nan).notna() & \
                        ts.replace([np.inf, -np.inf], np.nan).notna()
            if not bool(valid_now.any()): continue

            sub = pd.DataFrame({
                "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                "ts":     ts.loc[valid_now].astype("Int64").values.astype(np.int64),
                "a":      a.loc[valid_now].values.astype(np.float64),
                "b":      b.loc[valid_now].values.astype(np.float64),
            })
            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)

            p_ts = sub.groupby("symbol", sort=False)["ts"].shift(1)
            p_a  = sub.groupby("symbol", sort=False)["a"].shift(1)
            p_b  = sub.groupby("symbol", sort=False)["b"].shift(1)

            # 首行用跨批次状态填充
            first = p_ts.isna()
            if bool(first.any()):
                syms = sub.loc[first, "symbol"].values
                p_ts.loc[first] = np.array([last_ts_by.get(s) for s in syms], dtype="float64")
                # 直接用跨批次 is_locked，避免精度误差
                prev_locked_fill = np.array([last_is_locked_by.get(s, np.nan) for s in syms], dtype="float64")
                # 如果没有历史锁定状态，则需要基于 p_a/p_b（可能都是 NaN），后面统一判断
                # 这里只把 p_a/p_b 设为 NaN，让下一步逻辑通过 prev_locked_fill 覆盖
                p_a.loc[first] = np.nan
                p_b.loc[first] = np.nan

            # 计算上一事件是否锁定/交叉
            prev_locked = np.where(np.isfinite(p_a.values) & np.isfinite(p_b.values),
                                   (p_a.values <= p_b.values).astype(float),
                                   np.nan)

            # 用跨批次布尔状态补缺
            if bool(first.any()):
                idx = np.where(first.values)[0]
                for j, irow in enumerate(idx):
                    if np.isfinite(prev_locked[irow]):
                        continue
                    fill = last_is_locked_by.get(sub.loc[irow, "symbol"], None)
                    prev_locked[irow] = float(1.0 if fill else 0.0) if (fill is not None) else np.nan

            # 与 RTH 交叠时长
            t0 = p_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left  = np.maximum(t0, float(T0))
            right = np.minimum(t1, float(T1))
            dt_ns = np.clip(right - left, 0.0, None)

            valid_pair = (dt_ns > 0.0) & np.isfinite(prev_locked)
            if bool(valid_pair.any()):
                tmp = pd.DataFrame({
                    "symbol": sub["symbol"].values,
                    "t":  np.where(valid_pair, dt_ns, 0.0),
                    "lk": np.where(valid_pair, prev_locked, 0.0),  # 0/1
                }).groupby("symbol", observed=True).sum()

                for sym, row in tmp.iterrows():
                    time_ns_by[sym] = time_ns_by.get(sym, 0.0) + float(row["t"])
                    lock_ns_by[sym] = lock_ns_by.get(sym, 0.0) + float(row["t"] * row["lk"])

            # 更新跨批次
            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r in tail.iterrows():
                last_ts_by[r["symbol"]] = int(r["ts"])
                last_is_locked_by[r["symbol"]] = bool(r["a"] <= r["b"])

        # 尾段补时长
        for sym, ts_last in last_ts_by.items():
            dt_ns = max(0.0, float(T1) - max(float(ts_last), float(T0)))
            if dt_ns <= 0:
                continue
            time_ns_by[sym] = time_ns_by.get(sym, 0.0) + dt_ns
            if last_is_locked_by.get(sym, False):
                lock_ns_by[sym] = lock_ns_by.get(sym, 0.0) + dt_ns

        out = sample[["symbol"]].copy()
        vals = []
        for sym in out["symbol"].astype(str).values:
            denom = time_ns_by.get(sym, 0.0)
            if denom > 0:
                vals.append(lock_ns_by.get(sym, 0.0) / denom)
            else:
                vals.append(pd.NA)
        out["value"] = vals
        return out


feature = QuoteLockedCrossedShareTWOnefileFeature()
