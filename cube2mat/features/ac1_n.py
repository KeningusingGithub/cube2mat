# features/ac1_n.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class AC1NFeature(BaseFeature):
    """
    09:30–15:59 内，成交笔数的 lag-1 自相关。
    """

    name = "ac1_n"
    description = "Lag-1 autocorrelation of trade counts n across intraday bars."
    required_full_columns = ("symbol", "time", "n")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _ac1(x: pd.Series) -> float:
        if len(x) < 2:
            return np.nan
        x = x.astype(float)
        x0 = x.iloc[:-1]
        x1 = x.iloc[1:]
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

        df = df.copy()
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["n"])
        if df.empty:
            out["value"] = pd.NA
            return out

        value = df.sort_index().groupby("symbol")["n"].apply(self._ac1)
        out["value"] = out["symbol"].map(value)
        return out


feature = AC1NFeature()
