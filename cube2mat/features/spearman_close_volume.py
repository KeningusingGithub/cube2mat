# features/spearman_close_volume.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class SpearmanCloseVolumeFeature(BaseFeature):
    """
    09:30–15:59 内，Spearman 等级相关：corr(rank(close), rank(volume))。
    若有效样本<2 或全同值导致秩方差为 0，则 NaN。
    """

    name = "spearman_close_volume"
    description = "Spearman rank correlation between close and volume within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
        xd = x - x.mean()
        yd = y - y.mean()
        sxx = (xd * xd).sum()
        syy = (yd * yd).sum()
        if sxx <= 0 or syy <= 0:
            return np.nan
        return float((xd * yd).sum() / np.sqrt(sxx * syy))

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

        def per_symbol(g: pd.DataFrame) -> float:
            if len(g) < 2:
                return np.nan
            rc = g["close"].rank(method="average")
            rv = g["volume"].rank(method="average")
            return self._pearson_corr(rc, rv)

        value = df.sort_index().groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = SpearmanCloseVolumeFeature()
