# features/time_share_above_vwap.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class TimeShareAboveVWAPFeature(BaseFeature):
    """
    收盘价高于 vwap 的时间占比： count(close>vwap)/count(valid pair)。
    """

    name = "time_share_above_vwap"
    description = "Fraction of bars with close > vwap during the session."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close", "vwap"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(sample["symbol"].unique())]

        for c in ("close", "vwap"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "vwap"])
        if df.empty:
            out["value"] = pd.NA
            return out

        value = (df["close"] > df["vwap"]).groupby(df["symbol"]).mean()
        out["value"] = out["symbol"].map(value.astype(float))
        return out


feature = TimeShareAboveVWAPFeature()
