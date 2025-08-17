# features/vwap_cross_spacing_mean.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

TOT_MIN = 389.0


class VWAPCrossSpacingMeanFeature(BaseFeature):
    """
    Mean time spacing (in fraction of session minutes) between consecutive crossings of (close - vwap).
    Steps:
      - consider only non-zero sign points
      - crossing timestamps where sign changes
      - spacing = mean(minute gaps)/389; NaN if <2 crossings
    """
    name = "vwap_cross_spacing_mean"
    description = "Mean spacing between VWAP crossings as fraction of 389 minutes; NaN if <2 crossings."
    required_full_columns = ("symbol", "time", "close", "vwap")
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

        for col in ("close","vwap"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close","vwap"])
        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
        if df.empty: out["value"] = pd.NA; return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            d = (g["close"] - g["vwap"]).to_numpy()
            s = np.sign(d)
            mask = s != 0
            if mask.sum() < 2:
                res[sym] = np.nan; continue
            s_nz = s[mask]
            t_nz = g.index.values[mask]
            cross_idx = np.nonzero(s_nz[1:] != s_nz[:-1])[0] + 1
            if cross_idx.size < 2:
                res[sym] = np.nan; continue
            t = pd.DatetimeIndex(t_nz)
            start = self._start(t)
            mins = ((t[cross_idx] - start).total_seconds()/60.0).astype(float)
            gaps = np.diff(mins)
            val = float(np.mean(gaps)/TOT_MIN) if gaps.size > 0 else np.nan
            res[sym] = val
        out["value"] = out["symbol"].map(res)
        return out


feature = VWAPCrossSpacingMeanFeature()
