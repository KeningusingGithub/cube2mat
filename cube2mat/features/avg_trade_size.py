# features/avg_trade_size.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class AvgTradeSizeFeature(BaseFeature):
    """
    Average shares per trade between 09:30–15:59:
        value = sum(volume) / sum(n); NaN if sum(n)<=0.
    """
    name = "avg_trade_size"
    description = "Average shares per trade: sum(volume)/sum(n) within 09:30–15:59; NaN if no trades."
    required_full_columns = ("symbol", "time", "volume", "n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty:
            out["value"] = pd.NA; return out

        df = self.ensure_et_index(df_full, "time", ctx.tz).between_time("09:30", "15:59")
        if df.empty:
            out["value"] = pd.NA; return out

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))].copy()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["volume","n"])
        if df.empty:
            out["value"] = pd.NA; return out

        agg = df.groupby("symbol").agg(vol=("volume","sum"), trades=("n","sum"))
        agg["value"] = np.where(agg["trades"]>0, agg["vol"]/agg["trades"], np.nan)

        out["value"] = out["symbol"].map(agg["value"])
        return out


feature = AvgTradeSizeFeature()
