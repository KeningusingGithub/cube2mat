# features/time_to_20pct_volume_min.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class TimeTo20PctVolumeMinFeature(BaseFeature):
    """
    09:30–15:59 内，累积成交量达到 20% 所需时间占全日交易时长的比例（自 09:30 起）。
    若 sum(vol)<=0 则 NaN。
    """

    name = "time_to_20pct_volume_min"
    description = (
        "Fraction of trading session required to reach 20% of cumulative volume."
    )
    required_full_columns = ("symbol", "time", "volume")
    required_pv_columns = ("symbol",)

    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

    @staticmethod
    def _minutes_since_open(idx: pd.DatetimeIndex) -> np.ndarray:
        return (
            (
                idx - idx.normalize() - pd.Timedelta("09:30:00")
            ).total_seconds()
            / 60.0
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

        df = df.copy()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["volume"])
        if df.empty:
            out["value"] = pd.NA
            return out
        df = df.sort_index()

        def per_symbol(g: pd.DataFrame) -> float:
            v = g["volume"].astype(float)
            tot = v.sum()
            if not np.isfinite(tot) or tot <= 0:
                return np.nan
            c = v.cumsum().values
            thr = 0.2 * tot
            i = int(np.searchsorted(c, thr, side="left"))
            i = min(i, len(g) - 1)
            tmins = self._minutes_since_open(g.index)
            return float(tmins[i] / self.TOTAL_MIN)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = TimeTo20PctVolumeMinFeature()
