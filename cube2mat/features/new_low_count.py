# features/new_low_count.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class NewLowCountFeature(BaseFeature):
    """Count of times a new session LOW is set in 09:30–15:59."""

    name = "new_low_count"
    description = "Number of new intraday lows within 09:30–15:59."
    required_full_columns = ("symbol", "time", "low")
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
        df = df[df["symbol"].isin(sample["symbol"].unique())].copy()
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df = df.dropna(subset=["low"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            l = g.sort_index()["low"]
            if len(l) < 2:
                res[sym] = np.nan
                continue
            prev_cum = l.shift(1).cummin()
            res[sym] = float(((l < prev_cum) & prev_cum.notna()).sum())
        out["value"] = out["symbol"].map(res)
        return out


feature = NewLowCountFeature()
