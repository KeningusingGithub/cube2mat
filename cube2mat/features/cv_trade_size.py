# features/cv_trade_size.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class CVTradeSizeFeature(BaseFeature):
    """
    Coefficient of variation of per-bar average trade size ts = volume / n
    using bars with n > 0 within RTH.
    """

    name = "cv_trade_size"
    description = "Std/mean of per-bar trade size (volume/n) in RTH."
    required_full_columns = ("symbol", "time", "volume", "n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "volume", "n"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59").copy()
        for c in ("volume", "n"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["volume", "n"])
        df = df[df["n"] > 0]
        df = df[df.symbol.isin(sample.symbol.unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        ts = df["volume"] / df["n"]
        res = df.assign(ts=ts).groupby("symbol")["ts"].apply(
            lambda s: float(s.std(ddof=1) / s.mean()) if (len(s) >= 3 and s.mean() > 0) else np.nan
        )
        out["value"] = out["symbol"].map(res)
        return out


feature = CVTradeSizeFeature()
