# features/close_on_volume_beta.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class CloseOnVolumeBetaFeature(BaseFeature):
    """
    09:30–15:59 内，按 symbol 做 OLS: close ~ volume，输出斜率（价格/股）。
    若有效样本<2 或 var(volume)=0，则 NaN。
    """

    name = "close_on_volume_beta"
    description = (
        "OLS slope of close on volume within 09:30–15:59; NaN if insufficient or var(volume)=0."
    )
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _ols_slope(x: pd.Series, y: pd.Series) -> float:
        n = len(x)
        if n < 2:
            return np.nan
        xm = x.mean()
        ym = y.mean()
        xd = x - xm
        yd = y - ym
        sxx = (xd * xd).sum()
        if not np.isfinite(sxx) or sxx <= 0:
            return np.nan
        sxy = (xd * yd).sum()
        return float(sxy / sxx)

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
            lambda g: self._ols_slope(g["volume"], g["close"])
        )
        out["value"] = out["symbol"].map(value)
        return out


feature = CloseOnVolumeBetaFeature()
