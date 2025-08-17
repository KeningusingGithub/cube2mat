# features/ac1_absret.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class AC1AbsRetFeature(BaseFeature):
    """
    09:30–15:59 内，|log 收益| 的 lag-1 皮尔逊自相关；len<2 或方差为0 则 NaN。
    """

    name = "ac1_absret"
    description = "Lag-1 autocorrelation of |log returns| (volatility clustering)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _ac1(x: pd.Series) -> float:
        if len(x) < 2:
            return np.nan
        x0 = x.iloc[:-1].astype(float)
        x1 = x.iloc[1:].astype(float)
        xd0 = x0 - x0.mean()
        xd1 = x1 - x1.mean()
        s00 = (xd0 * xd0).sum()
        s11 = (xd1 * xd1).sum()
        if s00 <= 0 or s11 <= 0:
            return np.nan
        return float((xd0 * xd1).sum() / np.sqrt(s00 * s11))

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

        df["absr"] = df["r"].abs()
        value = df.groupby("symbol")["absr"].apply(self._ac1)
        out["value"] = out["symbol"].map(value)
        return out


feature = AC1AbsRetFeature()

