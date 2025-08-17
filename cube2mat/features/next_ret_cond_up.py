# features/next_ret_cond_up.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class NextRetCondUpFeature(BaseFeature):
    """
    条件期望：E[ret_{t+1} | ret_t > 0]。
    用 simple return，过滤 inf/NaN；触发样本<3 则 NaN。
    """

    name = "next_ret_cond_up"
    description = "Mean of next simple return conditional on current ret>0."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

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
        if df.empty:
            out["value"] = pd.NA
            return out

        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df["ret_next"] = df.groupby("symbol", sort=False)["ret"].shift(-1)
        df = df.dropna(subset=["ret", "ret_next"])

        def per_symbol(g: pd.DataFrame) -> float:
            x = g.loc[g["ret"] > 0, "ret_next"]
            return float(x.mean()) if x.size >= 3 else np.nan

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = NextRetCondUpFeature()
