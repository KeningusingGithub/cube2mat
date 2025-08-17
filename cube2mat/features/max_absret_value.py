# features/max_absret_value.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class MaxAbsRetValueFeature(BaseFeature):
    """Max absolute simple return within 09:30–15:59."""

    name = "max_absret_value"
    description = "Session max |simple return| in RTH."
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
            r = (
                g.sort_index()["close"].pct_change()
                .replace([np.inf, -np.inf], np.nan)
                .abs()
                .dropna()
            )
            res[sym] = float(r.max()) if len(r) >= 1 else np.nan
        out["value"] = out["symbol"].map(res)
        return out


feature = MaxAbsRetValueFeature()
