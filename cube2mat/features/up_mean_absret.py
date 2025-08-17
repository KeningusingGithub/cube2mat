# features/up_mean_absret.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class UpMeanAbsRetFeature(BaseFeature):
    """
    上涨状态（ret>0）下 |ret| 的均值。有效上涨样本<1 则 NaN。
    """

    name = "up_mean_absret"
    description = "Mean absolute simple return conditional on ret>0."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59").sort_index()
        df = df[df["symbol"].isin(sample["symbol"].unique())]

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out
        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret"])

        def per_symbol(g: pd.DataFrame) -> float:
            x = g.loc[g["ret"] > 0, "ret"].abs()
            return float(x.mean()) if x.size > 0 else np.nan

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = UpMeanAbsRetFeature()
