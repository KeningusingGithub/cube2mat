# features/post_range_hl.py
from __future__ import annotations
import datetime as dt
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class PostRangeHLFeature(BaseFeature):
    """
    盘后 16:00–23:59 内，按 symbol 计算 max(high) - min(low)。
    输出: ['symbol', 'value']。
    """
    name = "post_range_hl"
    description = "After-hours range: max(high) - min(low) between 16:00–23:59."
    required_full_columns = ("symbol", "time", "high", "low")
    required_pv_columns   = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None

        if df_full.empty or sample.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        df_full = self.ensure_et_index(df_full, time_col="time", tz=ctx.tz)
        df = df_full.between_time("16:00", "23:59")
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        syms = set(sample["symbol"].unique())
        df = df[df["symbol"].isin(syms)]
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        df = df.copy()
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"]  = pd.to_numeric(df["low"],  errors="coerce")
        df = df.dropna(subset=["high", "low"])
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        hi = df.groupby("symbol")["high"].max()
        lo = df.groupby("symbol")["low"].min()
        val = hi - lo

        out = sample[["symbol"]].copy()
        out["value"] = out["symbol"].map(val)
        return out

feature = PostRangeHLFeature()
