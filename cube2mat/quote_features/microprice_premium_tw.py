# quote_features/microprice_premium_tw.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteMicroPremiumTWOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','ask_price','bid_price','ask_size','bid_size','participant_timestamp']，单次流式扫描整天，
    计算 RTH(09:30–16:00 ET) 的“时间加权微价溢价” ((micro - mid) / mid)。
    采用 piecewise-constant：以上一次事件的值持有到下一事件；最后补到 16:00。
    输出：['symbol','value'] 按 PV 样本顺序对齐。
    """

    name = "quote_micro_premium_tw_all"
    description = "RTH time-weighted mean of microprice premium per symbol (onefile, single pass)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    # 配置（如无特殊需要，不必改）
    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = (
        "ask_price",
        "bid_price",
        "ask_size",
        "bid_size",
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
        # 1) PV 样本
        sample = self.load_pv(ctx, date, columns=["symbol"])
        if sample is None:
            return None
        if sample.empty:
            return pd.DataFrame(columns=["symbol", "value"])

        # 2) 当日 onefile
        root = Path(getattr(ctx, "quote_root", self.default_quote_root))
        day_path = root / f"{date.strftime('%Y%m%d')}.parquet"
        if not day_path.exists():
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        # 3) 单次流式扫描并时间加权
        tz_name = getattr(ctx, "tz", "America/New_York")
        rth_start_ns, rth_end_ns = self._rth_bounds_utc_ns(date, tz_name, self.RTH_START, self.RTH_END)

        pf = pq.ParquetFile(str(day_path))
        cols = [
            "symbol",
            "ask_price",
            "bid_price",
            "ask_size",
            "bid_size",
            "participant_timestamp",
        ]

        w_by: Dict[str, float] = {}
        ws_by: Dict[str, float] = {}
        last_ts_by: Dict[str, int] = {}
        last_val_by: Dict[str, float] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            a = pd.to_numeric(df["ask_price"], errors="coerce")
            b = pd.to_numeric(df["bid_price"], errors="coerce")
            asz = pd.to_numeric(df["ask_size"], errors="coerce")
            bsz = pd.to_numeric(df["bid_size"], errors="coerce")
            ts = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            size_sum = asz + bsz
            mid = (a + b) / 2.0

            valid_now = (
                a.replace([np.inf, -np.inf], np.nan).notna()
                & b.replace([np.inf, -np.inf], np.nan).notna()
                & asz.replace([np.inf, -np.inf], np.nan).notna()
                & bsz.replace([np.inf, -np.inf], np.nan).notna()
                & ts.replace([np.inf, -np.inf], np.nan).notna()
                & size_sum.replace([np.inf, -np.inf], np.nan).notna()
                & (size_sum > 0.0)
                & mid.replace([np.inf, -np.inf], np.nan).notna()
                & (mid > 0.0)
            )
            if not bool(valid_now.any()):
                continue

            micro = (a * bsz + b * asz) / size_sum
            premium = ((micro - mid) / mid).astype(float)

            sub = pd.DataFrame(
                {
                    "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                    "ts": ts.loc[valid_now].astype("Int64").values.astype(np.int64),
                    "val": premium.loc[valid_now].values.astype(np.float64),
                }
            )

            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)

            prev_ts = sub.groupby("symbol", sort=False)["ts"].shift(1)
            prev_val = sub.groupby("symbol", sort=False)["val"].shift(1)

            first_mask = prev_ts.isna()
            if bool(first_mask.any()):
                syms_first = sub.loc[first_mask, "symbol"].values
                fill_ts = np.array([last_ts_by.get(s) for s in syms_first], dtype="float64")
                fill_val = np.array([last_val_by.get(s) for s in syms_first], dtype="float64")
                prev_ts.loc[first_mask] = fill_ts
                prev_val.loc[first_mask] = fill_val

            t0 = prev_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left = np.maximum(t0, float(rth_start_ns))
            right = np.minimum(t1, float(rth_end_ns))
            dt_ns = np.clip(right - left, 0.0, None)

            w = pd.Series(dt_ns, index=sub.index)
            ws = w * prev_val.values

            agg = (
                pd.DataFrame({"symbol": sub["symbol"].values, "w": w.values, "ws": ws.values})
                .groupby("symbol", observed=True)
                .agg(w_sum=("w", "sum"), ws_sum=("ws", "sum"))
            )

            for sym, row in agg.iterrows():
                w_by[sym] = w_by.get(sym, 0.0) + float(row["w_sum"])
                ws_by[sym] = ws_by.get(sym, 0.0) + float(row["ws_sum"])

            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r in tail.iterrows():
                last_ts_by[r["symbol"]] = int(r["ts"])
                last_val_by[r["symbol"]] = float(r["val"])

        # 收尾：补到 16:00
        for sym, ts_last in last_ts_by.items():
            val_last = last_val_by.get(sym, np.nan)
            if not np.isfinite(ts_last) or not np.isfinite(val_last):
                continue
            left = max(float(ts_last), float(rth_start_ns))
            right = float(rth_end_ns)
            dt_ns = max(0.0, right - left)
            if dt_ns > 0:
                w_by[sym] = w_by.get(sym, 0.0) + dt_ns
                ws_by[sym] = ws_by.get(sym, 0.0) + dt_ns * float(val_last)

        mean_by = {k: (ws_by[k] / w) for k, w in w_by.items() if w > 0.0}

        out = sample[["symbol"]].copy()
        out["value"] = [mean_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteMicroPremiumTWOnefileFeature()
