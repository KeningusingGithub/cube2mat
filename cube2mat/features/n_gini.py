# cube2mat/features/n_gini.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class NGiniFeature(BaseFeature):
    """
    Gini index for distribution of n across intraday bars (09:30–15:59).
    Use standard discrete Gini on nonnegative values; NaN if sum<=0 or <2 bars.
    """
    name = "n_gini"
    description = "Gini index of trade count distribution across RTH bars."
    required_full_columns = ("symbol", "time", "n")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _gini(x: np.ndarray) -> float:
        x = x.astype(float)
        x = x[np.isfinite(x) & (x >= 0)]
        n = x.size
        s = x.sum()
        if n < 2 or s <= 0:
            return np.nan
        xs = np.sort(x)
        cum = np.cumsum(xs)
        # Gini = 1 - 2 * sum((n - i + 0.5) * x_i) / (n * sum(x))
        i = np.arange(1, n + 1)
        g = 1.0 - 2.0 * np.sum((n - i + 0.5) * xs) / (n * s)
        return float(np.clip(g, 0.0, 1.0))

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, list(self.required_pv_columns))
        if df_full is None or sample is None: return None
        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty: out["value"] = pd.NA; return out

        df = self.ensure_et_index(df_full, "time", ctx.tz).between_time("09:30","15:59")
        if df.empty: out["value"] = pd.NA; return out

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))].copy()
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["n"])
        if df.empty: out["value"] = pd.NA; return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            res[sym] = self._gini(g.sort_index()["n"].to_numpy())
        out["value"] = out["symbol"].map(res)
        return out

feature = NGiniFeature()