# features/trend_piecewise_slope_am_pm.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class TrendPiecewiseSlopeAmPmFeature(BaseFeature):
    """Difference between afternoon and morning linear slopes."""

    name = "trend_piecewise_slope_am_pm"
    description = "Afternoon minus morning OLS slope of close~time within RTH."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def _ols_slope(self, y: pd.Series) -> float:
        n = y.size
        if n < 2:
            return np.nan
        t = np.linspace(0.0, 1.0, n, endpoint=True)
        x = t - t.mean()
        y_d = y.to_numpy(dtype=float) - y.mean()
        den = (x * x).sum()
        if den <= 0:
            return np.nan
        return float((x * y_d).sum() / den)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
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
        df = df.dropna(subset=["close"]).sort_index()
        if df.empty:
            out["value"] = pd.NA
            return out

        am = df.between_time("09:30", "12:00")
        pm = df.between_time("12:00", "15:59")

        res = {}
        for sym in sample["symbol"].unique():
            ya = am[am["symbol"] == sym]["close"]
            yp = pm[pm["symbol"] == sym]["close"]
            if ya.size < 2 or yp.size < 2:
                res[sym] = np.nan
                continue
            res[sym] = self._ols_slope(yp) - self._ols_slope(ya)
        out["value"] = out["symbol"].map(res)
        return out


feature = TrendPiecewiseSlopeAmPmFeature()
