from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteStaleQuoteShareOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','sip_timestamp','participant_timestamp']，单次流式扫描，
    计算 RTH 内“陈旧报价占比”（SIP 时间轴）：
      设阈值 λ（毫秒，默认 ctx.stale_gap_ms 或 500ms），在任意时刻 t，若上一次 SIP 更新距今的间隔 > λ，则视为陈旧。
      令 RTH=[T0,T1)，给定某 symbol 的 SIP 更新时刻序列 {t0<t1<...}，则陈旧时长为
          ∑ max(0, min(t_{i+1}, T1) - max(t_i + λ, T0))，
        另加首尾段：若首个更新 t_first>T0，额外计入 max(0, min(t_first, T1) - (T0 + λ))；
                     尾段为 max(0, T1 - max(t_last + λ, T0))。
    占比 = 陈旧时长 / (T1 - T0)。输出：['symbol','value'] ∈ [0,1]。
    """
    name = "quote_stale_quote_share_all"
    description = "RTH time share (on SIP timeline) where age since last SIP update exceeds λ (onefile)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000
    DEFAULT_LAMBDA_MS = 500.0

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
        T0, T1 = self._rth_bounds_utc_ns(date, tz_name, self.RTH_START, self.RTH_END)
        lam_ms = float(getattr(ctx, "stale_gap_ms", self.DEFAULT_LAMBDA_MS))
        lam_ns = lam_ms * 1e6

        pf = pq.ParquetFile(str(day_path))
        cols = ["symbol", "sip_timestamp"]

        # 累加器：陈旧时长（ns）；同时记录首末 SIP 时间
        stale_ns_by: Dict[str, float] = {}
        first_sip_by: Dict[str, int] = {}
        last_sip_by: Dict[str, int] = {}

        # 跨批次“上一 SIP 更新时间”
        prev_sip_by: Dict[str, int] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            sip = pd.to_numeric(df["sip_timestamp"], errors="coerce")
            valid = sip.replace([np.inf, -np.inf], np.nan).notna()
            if not bool(valid.any()):
                continue

            sub = pd.DataFrame(
                {
                    "symbol": df.loc[valid, "symbol"].astype(str).values,
                    "sip": sip.loc[valid].astype("Int64").values.astype(np.int64),
                }
            )

            # 按 (symbol, sip) 排序
            sub.sort_values(["symbol", "sip"], kind="mergesort", inplace=True)

            # 为每个 symbol 遍历
            for sym, g in sub.groupby("symbol", sort=False):
                arr = g["sip"].values.astype(np.int64)
                if arr.size == 0:
                    continue

                # 初始化首尾
                if sym not in first_sip_by:
                    first_sip_by[sym] = int(arr[0])
                # 处理跨批次“上一 sip”
                prev = prev_sip_by.get(sym, None)

                for cur in arr:
                    if prev is not None:
                        # 本段：[prev, cur)，在 RTH 内 age>λ 的时长
                        left = max(prev + lam_ns, T0)
                        right = min(cur, T1)
                        dt = max(0.0, right - left)
                        if dt > 0:
                            stale_ns_by[sym] = stale_ns_by.get(sym, 0.0) + dt
                    prev = int(cur)

                prev_sip_by[sym] = prev
                last_sip_by[sym] = prev

        # 首尾补偿：首段（T0 -> first_sip）与尾段（last_sip -> T1）
        for sym in set(list(first_sip_by.keys()) + list(last_sip_by.keys()) + list(prev_sip_by.keys())):
            first = first_sip_by.get(sym, None)
            last = last_sip_by.get(sym, None)

            # 若首个更新 > T0，则首段陈旧时长
            if first is not None and first > T0:
                left = T0 + lam_ns
                right = min(first, T1)
                dt = max(0.0, right - left)
                if dt > 0:
                    stale_ns_by[sym] = stale_ns_by.get(sym, 0.0) + dt

            # 尾段：从 last 到 T1
            if last is not None:
                left = max(last + lam_ns, T0)
                right = T1
                dt = max(0.0, right - left)
                if dt > 0:
                    stale_ns_by[sym] = stale_ns_by.get(sym, 0.0) + dt
            else:
                # 一整天无 SIP 更新：整个 RTH 都可能陈旧（按起点 T0 计）
                dt = max(0.0, T1 - (T0 + lam_ns))
                if dt > 0:
                    stale_ns_by[sym] = stale_ns_by.get(sym, 0.0) + dt

        total_ns = float(T1 - T0)
        share_by: Dict[str, float] = {}
        for sym in set(list(stale_ns_by.keys()) + list(first_sip_by.keys())):
            share_by[sym] = min(1.0, max(0.0, stale_ns_by.get(sym, 0.0) / total_ns))

        out = sample[["symbol"]].copy()
        out["value"] = [share_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteStaleQuoteShareOnefileFeature()
