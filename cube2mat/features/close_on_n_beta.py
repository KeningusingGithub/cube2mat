# features/close_on_n_beta.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class CloseOnNBetaFeature(BaseFeature):
    """
    09:30–15:59 内，OLS: close ~ n（成交笔数）。样本<2 或 var(n)=0 则 NaN。
    """

    name = "close_on_n_beta"
    description = "OLS slope of close on number of trades n within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "n")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _ols_slope(x: pd.Series, y: pd.Series) -> float:
        n = len(x)
        if n < 2:
            return np.nan
        xd = x - x.mean()
        yd = y - y.mean()
        sxx = (xd * xd).sum()
        if sxx <= 0:
            return np.nan
        return float(((xd * yd).sum()) / sxx)

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
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["close", "n"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        value = df.groupby("symbol").apply(
            lambda g: self._ols_slope(g["n"], g["close"])
        )
        out["value"] = out["symbol"].map(value)
        return out


feature = CloseOnNBetaFeature()
