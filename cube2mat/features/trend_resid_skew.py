# features/trend_resid_skew.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class TrendResidSkewFeature(BaseFeature):
    """Adjusted skewness of residuals from close ~ time (t in [0,1]) in RTH."""

    name = "trend_resid_skew"
    description = "Adjusted skewness of residuals from linear trend close~time in RTH."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def _lin_fit_resid(self, y: np.ndarray) -> np.ndarray:
        n = y.size
        t = np.linspace(0.0, 1.0, n, endpoint=True)
        X = np.column_stack([np.ones(n), t])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta

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
            try:
                e = self._lin_fit_resid(y)
            except Exception:
                res[sym] = np.nan
                continue
            m = e.mean()
            c2 = np.mean((e - m) ** 2)
            if c2 <= 0:
                res[sym] = np.nan
                continue
            c3 = np.mean((e - m) ** 3)
            g1 = c3 / (c2 ** 1.5)
            adj = np.sqrt(n * (n - 1)) / (n - 2) if n > 2 else np.nan
            res[sym] = float(adj * g1) if np.isfinite(adj) else np.nan
        out["value"] = out["symbol"].map(res)
        return out


feature = TrendResidSkewFeature()
