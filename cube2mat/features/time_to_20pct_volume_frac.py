# features/time_to_20pct_volume_frac.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class TimeTo20PctVolumeFracFeature(BaseFeature):
    """
    09:30–15:59 内，累积成交量达到 20% 所需的“时间占比”（分钟/总分钟，∈[0,1]）。
    若 sum(volume)<=0 则 NaN。
    """

    name = "time_to_20pct_volume_frac"
    description = "Fraction of session minutes to reach 20% cumulative volume."
    required_full_columns = ("symbol", "time", "volume")
    required_pv_columns = ("symbol",)
    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

    @staticmethod
    def _minutes_since_open(idx: pd.DatetimeIndex) -> np.ndarray:
        return (
            (idx - idx.normalize() - pd.Timedelta("09:30:00")).total_seconds() / 60.0
        ).astype(float)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, list(self.required_full_columns))
        sample = self.load_pv(ctx, date, list(self.required_pv_columns))
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

        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["volume"]).sort_index()
        if df.empty:
            out["value"] = pd.NA
            return out

        def per_symbol(g: pd.DataFrame) -> float:
            v = g["volume"].astype(float).values
            tot = v.sum()
            if not np.isfinite(tot) or tot <= 0 or len(v) == 0:
                return np.nan
            c = np.cumsum(v)
            thr = 0.2 * tot
            i = int(np.searchsorted(c, thr, side="left"))
            i = min(i, len(g) - 1)
            minutes = self._minutes_since_open(g.index)[i]
            return float(minutes / self.TOTAL_MIN)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = TimeTo20PctVolumeFracFeature()

