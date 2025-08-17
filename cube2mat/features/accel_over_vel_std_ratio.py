# features/accel_over_vel_std_ratio.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class AccelOverVelStdRatioFeature(BaseFeature):
    """
    Ratio of std(Δ²close) to std(Δclose) within RTH; measures curvature vs. velocity.
    Returns NaN if insufficient data or denominator is 0.
    """

    name = "accel_over_vel_std_ratio"
    description = "Std(second diff of close) / Std(first diff of close) in RTH."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59")
        df = df[df.symbol.isin(sample.symbol.unique())]
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            s = g.sort_index()["close"].to_numpy(dtype=float)
            if s.size < 4:
                res[sym] = np.nan
                continue
            d1 = np.diff(s)
            d2 = np.diff(d1)
            sd1 = float(np.std(d1, ddof=1)) if d1.size >= 2 else np.nan
            sd2 = float(np.std(d2, ddof=1)) if d2.size >= 2 else np.nan
            res[sym] = (sd2 / sd1) if (np.isfinite(sd1) and sd1 > 0 and np.isfinite(sd2)) else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = AccelOverVelStdRatioFeature()
