# features/breakout_nextret_high.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class BreakoutNextRetHighFeature(BaseFeature):
    """Mean next simple return conditional on a 'new high' breakout event."""

    name = "breakout_nextret_high"
    description = "Mean(next simple ret | new session high at t) within RTH."
    required_full_columns = ("symbol", "time", "high", "close")
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
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["high", "close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            prev_high = g["high"].shift(1).cummax()
            event = (g["high"] > prev_high) & prev_high.notna()
            next_ret = g["close"].pct_change().shift(-1)
            vals = next_ret[event].dropna()
            res[sym] = float(vals.mean()) if len(vals) >= 3 else np.nan
        out["value"] = out["symbol"].map(res)
        return out


feature = BreakoutNextRetHighFeature()
