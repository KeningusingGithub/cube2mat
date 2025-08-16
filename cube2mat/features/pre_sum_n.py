# features/pre_sum_n.py
from __future__ import annotations
import datetime as dt
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class PreSumNFeature(BaseFeature):
    """
    盘前 00:00–09:29 内，按 symbol 统计 n(number of trades) 总和。
    输出: ['symbol', 'value']。
    """
    name = "pre_sum_n"
    description = "Pre-market sum(n) 00:00–09:29."
    required_full_columns = ("symbol", "time", "n")
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
        df = df_full.between_time("00:00", "09:29")
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
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["n"])
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        val = df.groupby("symbol")["n"].sum()
        out = sample[["symbol"]].copy()
        out["value"] = out["symbol"].map(val)
        return out

feature = PreSumNFeature()
