# quote_features/opening_flag_share_tw.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple, Iterable, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


def _has_any_code(x: Iterable, code_set: Set[int]) -> bool:
    if not code_set or x is None:
        return False
    try:
        for v in x:
            if v in code_set:
                return True
    except TypeError:
        return False
    return False


class QuoteOpeningFlagShareTWOnefileFeature(QuoteBaseFeature):
    """
    在 RTH(09:30–16:00 ET) 内，时间加权统计 “开盘相关标志” 占比（上一事件在 [t_{i-1}, t_i) 持有）：
      share = time{ prev has opening-flag } / time{ 观测到的 RTH 时长 }
    标志由 conditions/indicators 是否包含 ctx 指定的代码集合决定：
      - opening_condition_codes: set[int]
      - opening_indicator_codes: set[int]
    若未提供集合则结果为 0（只计时长、分子始终 0）。
    """
    name = "quote_opening_flag_share_tw_all"
    description = "RTH time share where opening-related flags are set (conditions/indicators), piecewise-constant"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END   = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = ("conditions", "indicators", "participant_timestamp", "symbol")

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

        open_c = set(getattr(ctx, "opening_condition_codes", set()))
        open_i = set(getattr(ctx, "opening_indicator_codes", set()))
        tz = getattr(ctx, "tz", "America/New_York")
        T0, T1 = self._rth_bounds_utc_ns(date, tz, self.RTH_START, self.RTH_END)

        pf = pq.ParquetFile(str(path))
        cols = ["symbol", "conditions", "indicators", "participant_timestamp"]

        flag_ns_by: Dict[str, float] = {}
        time_ns_by: Dict[str, float] = {}

        # 上一事件状态
        last_ts_by: Dict[str, int] = {}
        last_flag_by: Dict[str, bool] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()
            ts = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            valid_now = ts.replace([np.inf, -np.inf], np.nan).notna()
            if not bool(valid_now.any()): continue

            sub = pd.DataFrame({
                "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                "ts":     ts.loc[valid_now].astype("Int64").values.astype(np.int64),
                "conditions": df.loc[valid_now, "conditions"],
                "indicators": df.loc[valid_now, "indicators"],
            })
            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)

            # 当前事件是否带“开盘标志”
            cur_flag = sub["conditions"].apply(lambda x: _has_any_code(x, open_c)) | \
                       sub["indicators"].apply(lambda x: _has_any_code(x, open_i))

            p_ts   = sub.groupby("symbol", sort=False)["ts"].shift(1)
            p_flag = sub.groupby("symbol", sort=False)[cur_flag.name].shift(1)  # 上一事件的 flag

            # 用跨批次状态填充首行
            first = p_ts.isna()
            if bool(first.any()):
                syms = sub.loc[first, "symbol"].values
                p_ts.loc[first]   = np.array([last_ts_by.get(s) for s in syms], dtype="float64")
                p_flag.loc[first] = np.array([last_flag_by.get(s, False) for s in syms], dtype="float64")

            # 与 RTH 的交叠
            t0 = p_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left  = np.maximum(t0, float(T0))
            right = np.minimum(t1, float(T1))
            dt_ns = np.clip(right - left, 0.0, None)

            valid_pair = (dt_ns > 0.0) & np.isfinite(p_flag.values)
            if bool(valid_pair.any()):
                tmp = pd.DataFrame({
                    "symbol": sub["symbol"].values,
                    "t":  np.where(valid_pair, dt_ns, 0.0),
                    "f":  np.where(valid_pair, p_flag.values.astype(float), 0.0),
                }).groupby("symbol", observed=True).sum()

                for sym, row in tmp.iterrows():
                    time_ns_by[sym] = time_ns_by.get(sym, 0.0) + float(row["t"])
                    flag_ns_by[sym] = flag_ns_by.get(sym, 0.0) + float(row["t"] * row["f"])

            # 更新跨批次状态（以“当前事件”的 flag 作为下一段持有）
            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r in tail.iterrows():
                sym = r["symbol"]
                last_ts_by[sym]   = int(r["ts"])
                # 需要重新计算“当前事件”的 flag
                lf = _has_any_code(r["conditions"], open_c) or _has_any_code(r["indicators"], open_i)
                last_flag_by[sym] = bool(lf)

        # 尾段补时长
        for sym, ts_last in last_ts_by.items():
            dt_ns = max(0.0, float(T1) - max(float(ts_last), float(T0)))
            if dt_ns <= 0:
                continue
            time_ns_by[sym] = time_ns_by.get(sym, 0.0) + dt_ns
            if last_flag_by.get(sym, False):
                flag_ns_by[sym] = flag_ns_by.get(sym, 0.0) + dt_ns

        out = sample[["symbol"]].copy()
        vals = []
        for sym in out["symbol"].astype(str).values:
            denom = time_ns_by.get(sym, 0.0)
            if denom > 0:
                vals.append(flag_ns_by.get(sym, 0.0) / denom)
            else:
                vals.append(pd.NA)
        out["value"] = vals
        return out


feature = QuoteOpeningFlagShareTWOnefileFeature()
