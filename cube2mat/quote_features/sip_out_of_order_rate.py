from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteSIPOutOfOrderRateOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet）。
    读取 ['symbol','sequence_number','sip_timestamp','participant_timestamp']，单次流式扫描，
    计算 RTH 内的“乱序率”（Out-of-Order via SIP）：
        对相邻事件对（与 RTH 有正交叠时长），若 Δsequence_number>0 且 Δsip_timestamp<=0，则记为乱序。
        乱序率 = #乱序 / #基数，其中 #基数 = # {Δsequence_number>0 且有正交叠时长}。
    输出：['symbol','value']（∈[0,1]）。
    """
    name = "quote_sip_out_of_order_rate_all"
    description = "RTH out-of-order rate: share of (Δseq>0 & Δsip<=0) among pairs with Δseq>0 (onefile)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = ("sequence_number", "sip_timestamp", "participant_timestamp", "symbol")

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
        cols = ["symbol", "sequence_number", "sip_timestamp", "participant_timestamp"]

        base_by: Dict[str, int] = {}
        ooo_by: Dict[str, int] = {}

        # 跨批次状态
        last_ts_by: Dict[str, int] = {}
        last_seq_by: Dict[str, float] = {}
        last_sip_by: Dict[str, int] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            seq = pd.to_numeric(df["sequence_number"], errors="coerce")
            sip = pd.to_numeric(df["sip_timestamp"], errors="coerce")
            par = pd.to_numeric(df["participant_timestamp"], errors="coerce")

            valid_now = (
                seq.replace([np.inf, -np.inf], np.nan).notna()
                & sip.replace([np.inf, -np.inf], np.nan).notna()
                & par.replace([np.inf, -np.inf], np.nan).notna()
            )
            if not bool(valid_now.any()):
                continue

            sub = pd.DataFrame(
                {
                    "symbol": df.loc[valid_now, "symbol"].astype(str).values,
                    "ts": par.loc[valid_now].astype("Int64").values.astype(np.int64),
                    "seq": seq.loc[valid_now].values.astype(np.float64),
                    "sip": sip.loc[valid_now].astype("Int64").values.astype(np.int64),
                }
            )

            # 排序 + 前值
            sub.sort_values(["symbol", "ts"], kind="mergesort", inplace=True)
            p_ts = sub.groupby("symbol", sort=False)["ts"].shift(1)
            p_seq = sub.groupby("symbol", sort=False)["seq"].shift(1)
            p_sip = sub.groupby("symbol", sort=False)["sip"].shift(1)

            # 跨批次填充
            first_mask = p_ts.isna()
            if bool(first_mask.any()):
                syms_first = sub.loc[first_mask, "symbol"].values
                p_ts.loc[first_mask] = np.array([last_ts_by.get(s) for s in syms_first], dtype="float64")
                p_seq.loc[first_mask] = np.array([last_seq_by.get(s) for s in syms_first], dtype="float64")
                p_sip.loc[first_mask] = np.array([last_sip_by.get(s) for s in syms_first], dtype="float64")

            # 与 RTH 的交叠时长（仅用于筛选“有效对”）
            t0 = p_ts.values.astype(np.float64)
            t1 = sub["ts"].values.astype(np.float64)
            left = np.maximum(t0, float(rth_start_ns))
            right = np.minimum(t1, float(rth_end_ns))
            dt_ns = np.clip(right - left, 0.0, None)

            # 差分
            dseq = sub["seq"].values - p_seq.values
            dsip = sub["sip"].values - p_sip.values

            base = (dt_ns > 0.0) & np.isfinite(dseq) & (dseq > 0.0) & np.isfinite(dsip)
            ooo = base & (dsip <= 0.0)

            if bool(base.any()):
                tmp = (
                    pd.DataFrame(
                        {
                            "symbol": sub["symbol"].values,
                            "b": np.where(base, 1, 0),
                            "o": np.where(ooo, 1, 0),
                        }
                    ).groupby("symbol", observed=True)
                ).sum()

                for sym, row in tmp.iterrows():
                    base_by[sym] = base_by.get(sym, 0) + int(row["b"])
                    ooo_by[sym] = ooo_by.get(sym, 0) + int(row["o"])

            # 更新跨批次状态
            tail = sub.groupby("symbol", sort=False).tail(1)
            for _, r in tail.iterrows():
                last_ts_by[r["symbol"]] = int(r["ts"])
                last_seq_by[r["symbol"]] = float(r["seq"])
                last_sip_by[r["symbol"]] = int(r["sip"])

        rate_by: Dict[str, float] = {}
        for sym, b in base_by.items():
            if b > 0:
                rate_by[sym] = ooo_by.get(sym, 0) / b

        out = sample[["symbol"]].copy()
        out["value"] = [rate_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteSIPOutOfOrderRateOnefileFeature()
