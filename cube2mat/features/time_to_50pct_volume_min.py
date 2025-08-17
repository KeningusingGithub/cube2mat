# features/time_to_50pct_volume_min.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class TimeTo50PctVolumeMinFeature(BaseFeature):
    """
    Minutes since 09:30 to reach 50% of cumulative volume within RTH.
    Returns NaN if total volume <= 0.
    """

    name = "time_to_50pct_volume_min"
    description = "Absolute minutes required to reach 50% of session volume."
    required_full_columns = ("symbol", "time", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "volume"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59")
        df = df[df.symbol.isin(sample.symbol.unique())].copy()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            total = float(g["volume"].sum())
            if total <= 0:
                res[sym] = np.nan
                continue
            cum = g["volume"].cumsum()
            idx = cum.searchsorted(0.5 * total)
            idx = int(min(idx, len(g) - 1))
            t = g.index[idx]
            start = g.index[0]
            minutes = float((t - start).total_seconds() / 60.0)
            res[sym] = minutes

        out["value"] = out["symbol"].map(res)
        return out


feature = TimeTo50PctVolumeMinFeature()
