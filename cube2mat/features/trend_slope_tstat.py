# features/trend_slope_tstat.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class TrendSlopeTstatFeature(BaseFeature):
    """
    09:30–15:59 内，OLS: close ~ time(分钟) 的斜率 t 统计量：
      t = beta1 / SE(beta1), 其中 SE(beta1) = sqrt( sigma^2 / Sxx ),
      sigma^2 = SSE/(n-2), Sxx = sum((t - mean(t))^2)。
    若 n<3 或 Sxx<=0 则 NaN。
    """
    name = "trend_slope_tstat"
    description = "t-stat of OLS slope for close~time (minutes since 09:30) within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _tstat(x: pd.Series, y: pd.Series) -> float:
        n = len(x)
        if n < 3:
            return np.nan
        xm = x.mean(); ym = y.mean()
        xd = x - xm; yd = y - ym
        sxx = (xd * xd).sum()
        if sxx <= 0:
            return np.nan
        beta1 = (xd * yd).sum() / sxx
        beta0 = ym - beta1 * xm
        resid = y - (beta0 + beta1 * x)
        sse = (resid * resid).sum()
        dof = n - 2
        if dof <= 0:
            return np.nan
        sigma2 = sse / dof
        if sigma2 <= 0:
            return np.nan
        se_beta1 = np.sqrt(sigma2 / sxx)
        if se_beta1 == 0:
            return np.nan
        return float(beta1 / se_beta1)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if full is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
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

        value = df.groupby("symbol").apply(lambda g: self._tstat(g["t_min"], g["close"]))
        out["value"] = out["symbol"].map(value)
        return out

feature = TrendSlopeTstatFeature()
