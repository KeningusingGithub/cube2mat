# features/near_high_time_share_10bp.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class NearHighTimeShare10bpFeature(BaseFeature):
    """Share of bars where close is within 10 bp of the session HIGH."""

    name = "near_high_time_share_10bp"
    description = "Fraction of bars with close near session HIGH within 10bp."
    required_full_columns = ("symbol", "time", "close", "high")
    required_pv_columns = ("symbol",)
    bp = 10

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
        for c in ("close", "high"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "high"])
        if df.empty:
            out["value"] = pd.NA
            return out

        thr = self.bp / 10000.0
        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            H = float(g["high"].max())
            if not np.isfinite(H) or H <= 0 or len(g) < 1:
                res[sym] = np.nan
                continue
            share = float(((H - g["close"]).clip(lower=0) / H <= thr).mean())
            res[sym] = share
        out["value"] = out["symbol"].map(res)
        return out


feature = NearHighTimeShare10bpFeature()
