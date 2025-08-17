# features/volume_center_of_mass_time.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VolumeCenterOfMassTimeFeature(BaseFeature):
    """Time center-of-mass (in [0,1]) using volume as weights."""

    name = "volume_center_of_mass_time"
    description = "Weighted time centroid by volume in RTH, normalized to [0,1]."
    required_full_columns = ("symbol", "time", "volume")
    required_pv_columns = ("symbol",)

    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

    def _start(self, idx: pd.DatetimeIndex) -> pd.Timestamp:
        day = idx[0].date()
        tz = idx.tz
        return pd.Timestamp.combine(day, dt.time(9, 30)).tz_localize(tz)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, list(self.required_full_columns))
        sample = self.load_pv(ctx, date, list(self.required_pv_columns))
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
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            if len(g) < 2:
                res[sym] = np.nan
                continue
            start = self._start(g.index)
            tfrac = ((g.index - start).total_seconds() / 60.0) / self.TOTAL_MIN
            w = g["volume"]
            wsum = float(w.sum())
            if not np.isfinite(wsum) or wsum <= 0:
                res[sym] = np.nan
                continue
            val = float((tfrac * w).sum() / wsum)
            res[sym] = float(np.clip(val, 0.0, 1.0))
        out["value"] = out["symbol"].map(res)
        return out


feature = VolumeCenterOfMassTimeFeature()
