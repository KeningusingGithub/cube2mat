# features/trend_slope_close_time.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class TrendSlopeCloseTimeFeature(BaseFeature):
    """
    09:30–15:59 内，对每个 symbol 做 OLS: close ~ time(分钟)。
    输出斜率（单位：价格/分钟）。n<2 或 var(time)=0 时 NaN。
    """
    name = "trend_slope_close_time"
    description = "OLS slope of close on minutes-since-09:30 within 09:30–15:59; NaN if <2 points or var(time)=0."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _ols_slope(x: pd.Series, y: pd.Series) -> float:
        n = len(x)
        if n < 2:
            return np.nan
        xm = x.mean(); ym = y.mean()
        xd = x - xm; yd = y - ym
        den = (xd * xd).sum()
        if not np.isfinite(den) or den <= 0:
            return np.nan
        num = (xd * yd).sum()
        return float(num / den)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df_full, time_col="time", tz=ctx.tz)
        df = df.between_time("09:30", "15:59")
        if df.empty:
            out["value"] = pd.NA
            return out

        syms = set(sample["symbol"].unique())
        df = df[df["symbol"].isin(syms)]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        # minutes since 09:30
        tdelta = (df.index - df.index.normalize()) - pd.Timedelta("09:30:00")
        df["t_min"] = tdelta.total_seconds() / 60.0

        value = (
            df.groupby("symbol")
              .apply(lambda g: self._ols_slope(g["t_min"], g["close"]))
        )

        out["value"] = out["symbol"].map(value)
        return out

feature = TrendSlopeCloseTimeFeature()
