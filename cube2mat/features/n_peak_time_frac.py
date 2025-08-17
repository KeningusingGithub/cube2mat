# features/n_peak_time_frac.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class NPeakTimeFracFeature(BaseFeature):
    """
    Fractional position (0..1) of the maximum n bar within the RTH sequence.
    Uses rank index (bar position), robust to irregular timestamps. NaN if <2 bars or max<=0.
    """
    name = "n_peak_time_frac"
    description = "Fractional bar-position when per-bar n peaks within 09:30–15:59."
    required_full_columns = ("symbol", "time", "n")
    required_pv_columns = ("symbol",)

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
            g = g.sort_index()
            m = len(g)
            if m < 2 or g["n"].max() <= 0:
                res[sym] = np.nan; continue
            pos = g["n"].values.argmax()
            res[sym] = float(pos / (m - 1)) if (m - 1) > 0 else np.nan
        out["value"] = out["symbol"].map(res)
        return out


feature = NPeakTimeFracFeature()
