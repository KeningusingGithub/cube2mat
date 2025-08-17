# features/ret_skew.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RetSkewFeature(BaseFeature):
    """
    09:30–15:59 内，log 收益的样本修正偏度：
      g1 = [n/((n-1)(n-2))] * sum((r-mean)^3) / s^3，s=样本标准差(ddof=1)。
    """

    name = "ret_skew"
    description = "Sample-adjusted skewness of intraday log returns."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _skew(s: pd.Series) -> float:
        r = s.values.astype(float)
        n = len(r)
        if n < 3:
            return np.nan
        m = r.mean()
        d = r - m
        s2 = np.sum(d * d) / (n - 1)
        if s2 <= 0:
            return np.nan
        s1 = np.sqrt(s2)
        m3 = np.sum(d ** 3) / n
        g1 = (n / ((n - 1) * (n - 2))) * (m3 / (s1 ** 3))
        return float(g1)

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

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df[(df["close"] > 0)].dropna(subset=["close"]).sort_index()
        if df.empty:
            out["value"] = pd.NA
            return out

        df["log_close"] = np.log(df["close"])
        df["r"] = df.groupby("symbol", sort=False)["log_close"].diff()
        df["r"] = df["r"].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["r"])
        if df.empty:
            out["value"] = pd.NA
            return out

        value = df.groupby("symbol")["r"].apply(self._skew)
        out["value"] = out["symbol"].map(value)
        return out


feature = RetSkewFeature()

