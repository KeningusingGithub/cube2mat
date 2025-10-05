# quote_features/tw_spread.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteTWSpreadOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    只读 ['symbol','ask_price','bid_price','participant_timestamp']，单次流式扫描整天，
    计算 RTH(09:30–16:00 ET) 的“时间加权相对价差” 2*(ask - bid)/(ask + bid)。
    采用 piecewise-constant：以上一次事件的值持有到下一事件；最后补到 16:00。
    对交叉/锁定报价 (ask<=bid) 截断为 0。
    输出：['symbol','value'] 按 PV 样本顺序对齐。
    """

    name = "quote_tw_spread_all"
    description = "RTH time-weighted mean of 2*(ask-bid)/(ask+bid) per symbol (onefile, single pass)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    # 配置（如无特殊需要，不必改）
    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = ("ask_price", "bid_price", "participant_timestamp", "symbol")

    @staticmethod
    def _rth_bounds_utc_ns(
        date: dt.date, tz_name: str, start: dt.time, end: dt.time
    ) -> Tuple[int, int]:
        """
        返回当日 [start, end) 在 UTC 的纳秒时间戳边界。
        """

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
        cols = ["symbol", "ask_price", "bid_price", "participant_timestamp"]

        # 累加器：∑(w), ∑(w*rel)；以及每个 symbol 的“上一事件状态”
        w_by: Dict[str, float] = {}
        ws_by: Dict[str, float] = {}
        last_ts_by: Dict[str, int] = {}
        last_rel_by: Dict[str, float] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            # 数值化 & 有效性
            a = pd.to_numeric(df["ask_price"], errors="coerce")
            b = pd.to_numeric(df["bid_price"], errors="coerce")
            denom = a + b
            ts = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            valid_now = (
                a.replace([np.inf, -np.inf], np.nan).notna()
                & b.replace([np.inf, -np.inf], np.nan).notna()
                & denom.replace([np.inf, -np.inf], np.nan).notna()
                & ts.replace([np.inf, -np.inf], np.nan).notna()
                & (denom > 0.0)
            )

            if not bool(valid_now.any()):
                continue

            # 当前事件的 rel，并对交叉/锁定截断为 0
            rel = (2.0 * (a - b) / denom).astype(float).clip(lower=0.0)

            sub = pd.DataFrame(
                {
                    "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                    "ts": ts.loc[valid_now].astype("Int64").values.astype(np.int64),
                    "rel": rel.loc[valid_now].values.astype(np.float64),
                }
            )

            # 按 (symbol, ts) 排序，便于构造前值
            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)

            # 前值（上一事件）的 ts/rel
            prev_ts = sub.groupby("symbol", sort=False)["ts"].shift(1)
            prev_rel = sub.groupby("symbol", sort=False)["rel"].shift(1)

            # 用跨批次的“上一次状态”填充每个 symbol 在本批的首行
            first_mask = prev_ts.isna()
            if bool(first_mask.any()):
                syms_first = sub.loc[first_mask, "symbol"].values
                fill_ts = np.array([last_ts_by.get(s) for s in syms_first], dtype="float64")
                fill_rel = np.array([last_rel_by.get(s) for s in syms_first], dtype="float64")
                prev_ts.loc[first_mask] = fill_ts
                prev_rel.loc[first_mask] = fill_rel

            # 计算与 RTH 的区间交叠时长 dt（单位：ns），并用“上一事件的 rel”加权
            t0 = prev_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left = np.maximum(t0, float(rth_start_ns))
            right = np.minimum(t1, float(rth_end_ns))
            dt_ns = np.clip(right - left, 0.0, None)

            w = pd.Series(dt_ns, index=sub.index)
            ws = w * prev_rel.values

            # 聚合本批的 ∑w 与 ∑(w*rel)
            agg = (
                pd.DataFrame({"symbol": sub["symbol"].values, "w": w.values, "ws": ws.values})
                .groupby("symbol", observed=True)
                .agg(w_sum=("w", "sum"), ws_sum=("ws", "sum"))
            )

            # 累加到全局
            for sym, row in agg.iterrows():
                w_by[sym] = w_by.get(sym, 0.0) + float(row["w_sum"])
                ws_by[sym] = ws_by.get(sym, 0.0) + float(row["ws_sum"])

            # 更新跨批次“上一次状态”
            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r in tail.iterrows():
                last_ts_by[r["symbol"]] = int(r["ts"])
                last_rel_by[r["symbol"]] = float(r["rel"])

        # 收尾：将最后一次事件的 rel 补到 16:00
        for sym, ts_last in last_ts_by.items():
            rel_last = last_rel_by.get(sym, np.nan)
            if not np.isfinite(ts_last) or not np.isfinite(rel_last):
                continue
            left = max(float(ts_last), float(rth_start_ns))
            right = float(rth_end_ns)
            dt_ns = max(0.0, right - left)
            if dt_ns > 0:
                w_by[sym] = w_by.get(sym, 0.0) + dt_ns
                ws_by[sym] = ws_by.get(sym, 0.0) + dt_ns * float(rel_last)

        # 4) 求时间加权均值并回填
        mean_by = {k: (ws_by[k] / w) for k, w in w_by.items() if w > 0.0}

        out = sample[["symbol"]].copy()
        out["value"] = [mean_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteTWSpreadOnefileFeature()
