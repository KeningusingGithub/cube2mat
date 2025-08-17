# features/n_peak_time_min.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

TOT_MIN = 389.0


class NPeakTimeMinFeature(BaseFeature):
    """
    Fraction of trading session elapsed when per-bar n reaches its maximum.
    Uses minutes from 09:30; result in [0,1]; NaN if all n<=0 or empty.
    """
    name = "n_peak_time_min"
    description = "Fraction of session elapsed (by minutes) when per-bar n peaks within 09:30–15:59."
    required_full_columns = ("symbol", "time", "n")
    required_pv_columns = ("symbol",)

    def _start(self, idx):
        day = idx[0].date(); tz = idx.tz
        return pd.Timestamp.combine(day, dt.time(9,30)).tz_localize(tz)

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
            if g["n"].max() <= 0 or g.empty:
                res[sym] = np.nan; continue
            idxmax = g["n"].idxmax()
            start = self._start(g.index)
            frac = ((idxmax - start).total_seconds() / 60.0) / TOT_MIN
            res[sym] = float(np.clip(frac, 0.0, 1.0))
        out["value"] = out["symbol"].map(res)
        return out


feature = NPeakTimeMinFeature()
