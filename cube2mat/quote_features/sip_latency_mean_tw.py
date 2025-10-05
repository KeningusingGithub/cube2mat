from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteSIPLatencyMeanTWOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','sip_timestamp','participant_timestamp']，单次流式扫描，
    计算 RTH(09:30–16:00 ET) 的“时间加权 SIP 延迟均值”（单位：毫秒）：
        latency_ms = (sip_timestamp - participant_timestamp) / 1e6
    采用 piecewise-constant：以上一事件的 latency 持有到下一事件；最后补到 16:00。
    输出：['symbol','value'] 与 PV 顺序对齐（ms）。
    """
    name = "quote_sip_latency_mean_tw_all"
    description = "RTH time-weighted mean of SIP latency (sip - participant) in ms per symbol (onefile, single pass)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = ("sip_timestamp", "participant_timestamp", "symbol")

    @staticmethod
    def _rth_bounds_utc_ns(date: dt.date, tz_name: str, start: dt.time, end: dt.time) -> Tuple[int, int]:
        start_local = pd.Timestamp(dt.datetime.combine(date, start)).tz_localize(tz_name)
        end_local = pd.Timestamp(dt.datetime.combine(date, end)).tz_localize(tz_name)
        return int(start_local.tz_convert("UTC").value), int(end_local.tz_convert("UTC").value)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        # 1) PV
        sample = self.load_pv(ctx, date, columns=["symbol"])
        if sample is None:
            return None
        if sample.empty:
            return pd.DataFrame(columns=["symbol", "value"])

        # 2) onefile
        root = Path(getattr(ctx, "quote_root", self.default_quote_root))
        day_path = root / f"{date.strftime('%Y%m%d')}.parquet"
        if not day_path.exists():
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        tz_name = getattr(ctx, "tz", "America/New_York")
        rth_start_ns, rth_end_ns = self._rth_bounds_utc_ns(date, tz_name, self.RTH_START, self.RTH_END)

        pf = pq.ParquetFile(str(day_path))
        cols = ["symbol", "sip_timestamp", "participant_timestamp"]

        # 累加器：∑w（ns）、∑(w*lat_ms)
        w_by: Dict[str, float] = {}
        ws_by: Dict[str, float] = {}
        # 跨批次“上一事件状态”（以 participant_timestamp 为持有时间轴）
        last_ts_by: Dict[str, int] = {}
        last_lat_ms_by: Dict[str, float] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            sip = pd.to_numeric(df["sip_timestamp"], errors="coerce")
            par = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            valid_now = (
                sip.replace([np.inf, -np.inf], np.nan).notna()
                & par.replace([np.inf, -np.inf], np.nan).notna()
            )
            if not bool(valid_now.any()):
                continue

            lat_ms = ((sip - par) / 1e6).astype(float)

            sub = pd.DataFrame(
                {
                    "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                    "ts": par.loc[valid_now].astype("Int64").values.astype(np.int64),
                    "val": lat_ms.loc[valid_now].values.astype(np.float64),
                }
            )

            # 排序 + 前值
            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)
            p_ts = sub.groupby("symbol", sort=False)["ts"].shift(1)
            p_val = sub.groupby("symbol", sort=False)["val"].shift(1)

            # 填入跨批次前值
            first_mask = p_ts.isna()
            if bool(first_mask.any()):
                syms_first = sub.loc[first_mask, "symbol"].values
                p_ts.loc[first_mask] = np.array([last_ts_by.get(s) for s in syms_first], dtype="float64")
                p_val.loc[first_mask] = np.array([last_lat_ms_by.get(s) for s in syms_first], dtype="float64")

            # 与 RTH 的交叠时长
            t0 = p_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left = np.maximum(t0, float(rth_start_ns))
            right = np.minimum(t1, float(rth_end_ns))
            dt_ns = np.clip(right - left, 0.0, None)

            w = pd.Series(dt_ns, index=sub.index)
            ws = w * p_val.values  # 以上一事件的延迟持有到当前

            # 聚合
            agg = (
                pd.DataFrame({"symbol": sub["symbol"].values, "w": w.values, "ws": ws.values})
                .groupby("symbol", observed=True)
                .agg(w_sum=("w", "sum"), ws_sum=("ws", "sum"))
            )
            for sym, row in agg.iterrows():
                w_by[sym] = w_by.get(sym, 0.0) + float(row["w_sum"])
                ws_by[sym] = ws_by.get(sym, 0.0) + float(row["ws_sum"])

            # 更新跨批次状态（最后一行）
            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r in tail.iterrows():
                last_ts_by[r["symbol"]] = int(r["ts"])
                last_lat_ms_by[r["symbol"]] = float(r["val"])

        # 收尾：补到 16:00
        for sym, ts_last in last_ts_by.items():
            val_last = last_lat_ms_by.get(sym, np.nan)
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


feature = QuoteSIPLatencyMeanTWOnefileFeature()
