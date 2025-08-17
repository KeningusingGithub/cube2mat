# features/volume_peak_time_frac.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VolumePeakTimeFracFeature(BaseFeature):
    """
    成交量峰值出现的“时间占比”（峰值出现的分钟 / 总分钟，∈[0,1]）。
    """

    name = "volume_peak_time_frac"
    description = "Fraction of session minutes when the per-bar max volume occurs."
    required_full_columns = ("symbol", "time", "volume")
    required_pv_columns = ("symbol",)
    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

    @staticmethod
    def _minutes_since_open(ts: pd.Timestamp) -> float:
        return float(
            (ts - ts.normalize() - pd.Timedelta("09:30:00")).total_seconds() / 60.0
        )

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
            idx = g["volume"].idxmax()
            if pd.isna(idx):
                return np.nan
            mins = self._minutes_since_open(idx)
            return float(mins / self.TOTAL_MIN)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = VolumePeakTimeFracFeature()

