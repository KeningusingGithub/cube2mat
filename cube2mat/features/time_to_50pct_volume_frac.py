# features/time_to_50pct_volume_frac.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class TimeTo50PctVolumeFracFeature(BaseFeature):
    """
    Fraction of RTH minutes to reach 50% cumulative volume; range [0, 1].
    Returns NaN if total volume <= 0.
    """

    name = "time_to_50pct_volume_frac"
    description = "Session fraction to reach 50% of cumulative volume."
    required_full_columns = ("symbol", "time", "volume")
    required_pv_columns = ("symbol",)

    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

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
            idx = int(min(g["volume"].cumsum().searchsorted(0.5 * total), len(g) - 1))
            t = g.index[idx]
            start = g.index[0]
            frac = float(((t - start).total_seconds() / 60.0) / self.TOTAL_MIN)
            res[sym] = float(np.clip(frac, 0.0, 1.0))

        out["value"] = out["symbol"].map(res)
        return out


feature = TimeTo50PctVolumeFracFeature()
