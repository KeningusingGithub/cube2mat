# features/trend_quad_beta1.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class TrendQuadBeta1Feature(BaseFeature):
    """OLS coefficient of t (linear term) in close ~ 1 + t + t^2, with t in [0,1]."""

    name = "trend_quad_beta1"
    description = "Linear coefficient (beta1) of quadratic trend close~1+t+t^2 in RTH."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
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
            y = g.sort_index()["close"].to_numpy(dtype=float)
            n = y.size
            if n < 3:
                res[sym] = np.nan
                continue
            t = np.linspace(0.0, 1.0, n, endpoint=True)
            X = np.column_stack([np.ones(n), t, t * t])
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                res[sym] = float(beta[1])
            except Exception:
                res[sym] = np.nan
        out["value"] = out["symbol"].map(res)
        return out


feature = TrendQuadBeta1Feature()
