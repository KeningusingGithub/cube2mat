# features/post_std_close.py
from __future__ import annotations
import datetime as dt
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class PostStdCloseFeature(BaseFeature):
    """
    盘后 16:00–23:59 内，按 symbol 计算 close 的标准差；若有效点 < 3，则 NaN。
    输出: ['symbol', 'value']。
    """
    name = "post_std_close"
    description = "After-hours std(close) between 16:00–23:59; NaN if <3 ticks."
    required_full_columns = ("symbol", "time", "close")
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
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        g = df.groupby("symbol")["close"]
        stats = g.agg(n="count", std="std")
        val = stats["std"].where(stats["n"] >= 3)

        out = sample[["symbol"]].copy()
        out["value"] = out["symbol"].map(val)
        return out

feature = PostStdCloseFeature()
