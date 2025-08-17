# features/corr_close_volume.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class CorrCloseVolumeFeature(BaseFeature):
    """
    09:30–15:59 内，Pearson 相关：corr(close, volume)。
    若样本<2 或方差为 0，则 NaN。
    """

    name = "corr_close_volume"
    description = "Pearson correlation between close and volume within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
        n = len(x)
        if n < 2:
            return np.nan
        xd = x - x.mean()
        yd = y - y.mean()
        sxx = (xd * xd).sum()
        syy = (yd * yd).sum()
        if sxx <= 0 or syy <= 0:
            return np.nan
        r = (xd * yd).sum() / np.sqrt(sxx * syy)
        return float(r)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        out = sample[["symbol"]].copy() if sample is not None else None
        if full is None or sample is None:
            return None
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.copy()
        for c in ("close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        value = df.groupby("symbol").apply(
            lambda g: self._pearson_corr(g["close"], g["volume"])
        )
        out["value"] = out["symbol"].map(value)
        return out


feature = CorrCloseVolumeFeature()
