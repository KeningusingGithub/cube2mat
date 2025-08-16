# features/post_sum_volume.py
from __future__ import annotations
import datetime as dt
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class PostSumVolumeFeature(BaseFeature):
    """
    盘后 16:00–23:59 内，按 symbol 统计 volume 总和。
    输出: ['symbol', 'value']。
    """
    name = "post_sum_volume"
    description = "After-hours sum(volume) 16:00–23:59."
    required_full_columns = ("symbol", "time", "volume")
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
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["volume"])
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        val = df.groupby("symbol")["volume"].sum()
        out = sample[["symbol"]].copy()
        out["value"] = out["symbol"].map(val)
        return out

feature = PostSumVolumeFeature()
