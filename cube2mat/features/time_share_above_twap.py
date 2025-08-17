# features/time_share_above_twap.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class TimeShareAboveTWAPFeature(BaseFeature):
    """Fraction of bars with close > TWAP (mean close) in 09:30–15:59."""

    name = "time_share_above_twap"
    description = "Share of bars where close > TWAP during RTH."
    required_full_columns = ("symbol", "time", "close")
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

        df = (
            self.ensure_et_index(full, "time", ctx.tz)
            .between_time("09:30", "15:59")
        )
        df = df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty:
            out["value"] = pd.NA
            return out
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            if len(g) < 2:
                res[sym] = np.nan
                continue
            twap = float(g["close"].mean())
            res[sym] = float((g["close"] > twap).mean())
        out["value"] = out["symbol"].map(res)
        return out


feature = TimeShareAboveTWAPFeature()
