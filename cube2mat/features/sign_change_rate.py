# features/sign_change_rate.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class SignChangeRateFeature(BaseFeature):
    """
    收益符号切换率：
      rate = count(sign(ret_t) != sign(ret_{t-1}) on nonzero pairs) / count(nonzero pairs)
    """

    name = "sign_change_rate"
    description = "Rate of sign flips between consecutive simple returns (excluding zeros)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _rate(r: pd.Series) -> float:
        s = np.sign(r.values)
        valid = (s[1:] != 0) & (s[:-1] != 0)
        if valid.sum() == 0:
            return np.nan
        flips = (s[1:][valid] != s[:-1][valid]).sum()
        return float(flips / valid.sum())

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59").sort_index()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(sample["symbol"].unique())]
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret"])

        value = df.groupby("symbol")["ret"].apply(self._rate)
        out["value"] = out["symbol"].map(value)
        return out


feature = SignChangeRateFeature()
