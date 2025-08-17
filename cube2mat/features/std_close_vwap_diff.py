# features/std_close_vwap_diff.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class StdCloseVWAPDiffFeature(BaseFeature):
    """
    09:30–15:59 内，（close - vwap）的样本标准差（ddof=1）。
    若有效样本<2 或任一列缺失，则 NaN。
    """

    name = "std_close_vwap_diff"
    description = "Sample std of (close - vwap) within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, list(self.required_full_columns))
        sample = self.load_pv(ctx, date, list(self.required_pv_columns))
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        df = df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        for c in ("close", "vwap"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "vwap"]).sort_index()
        if df.empty:
            out["value"] = pd.NA
            return out

        df["diff"] = df["close"] - df["vwap"]
        stats = df.groupby("symbol")["diff"].agg(n="count", std=lambda s: s.std(ddof=1))
        value = stats["std"].where(stats["n"] >= 2)
        out["value"] = out["symbol"].map(value)
        return out


feature = StdCloseVWAPDiffFeature()

