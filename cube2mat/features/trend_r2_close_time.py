# features/trend_r2_close_time.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class TrendR2CloseTimeFeature(BaseFeature):
    """
    09:30–15:59 内，OLS: close ~ time(分钟) 的拟合优度 R^2。
    若 var(close)=0 或样本<2，则 NaN。
    """
    name = "trend_r2_close_time"
    description = "R^2 of OLS close~time (minutes since 09:30) within 09:30–15:59; NaN if <2 points or var(close)=0."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _ols_r2(x: pd.Series, y: pd.Series) -> float:
        n = len(x)
        if n < 2:
            return np.nan
        xm = x.mean(); ym = y.mean()
        xd = x - xm; yd = y - ym
        den = (xd * xd).sum()
        sst = (yd * yd).sum()
        if den <= 0 or sst <= 0:
            return np.nan
        beta1 = (xd * yd).sum() / den
        ssr = (beta1 * beta1) * den
        r2 = ssr / sst
        return float(r2)

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

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
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
        tdelta = (df.index - df.index.normalize()) - pd.Timedelta("09:30:00")
        df["t_min"] = tdelta.total_seconds() / 60.0

        value = df.groupby("symbol").apply(lambda g: self._ols_r2(g["t_min"], g["close"]))
        out["value"] = out["symbol"].map(value)
        return out

feature = TrendR2CloseTimeFeature()
