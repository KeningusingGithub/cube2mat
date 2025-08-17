# features/vwap_cross_down_nextret.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VWAPCrossDownNextRetFeature(BaseFeature):
    """
    Mean next simple return conditional on downward crossing:
    event when (close - vwap) turns from >= 0 to < 0.
    Returns NaN if fewer than 3 events.
    """

    name = "vwap_cross_down_nextret"
    description = "Mean next ret after downward VWAP crossing in RTH."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "close", "vwap"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59").copy()
        for c in ("close", "vwap"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "vwap"])
        df = df[df.symbol.isin(sample.symbol.unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            d = g["close"] - g["vwap"]
            down = (d < 0) & (d.shift(1) >= 0)
            nxt = g["close"].pct_change().shift(-1)
            vals = nxt[down].dropna()
            res[sym] = float(vals.mean()) if len(vals) >= 3 else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = VWAPCrossDownNextRetFeature()
