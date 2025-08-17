# features/std_trade_size.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class StdTradeSizeFeature(BaseFeature):
    """
    Std of per-bar average trade size (volume/n) within 09:30–15:59.
    Exclude bars with n<=0; NaN if <3 valid bars.
    """
    name = "std_trade_size"
    description = "Std of per-bar trade size (volume/n) within 09:30–15:59; NaN if <3 valid bars."
    required_full_columns = ("symbol", "time", "volume", "n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty:
            out["value"] = pd.NA; return out

        df = self.ensure_et_index(df_full, "time", ctx.tz).between_time("09:30","15:59")
        if df.empty:
            out["value"] = pd.NA; return out

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))].copy()
        for col in ("volume","n"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["volume","n"])
        if df.empty:
            out["value"] = pd.NA; return out

        df = df[df["n"] > 0]
        if df.empty:
            out["value"] = pd.NA; return out

        df["tsize"] = df["volume"] / df["n"]
        res = df.groupby("symbol")["tsize"].agg(lambda s: float(s.std(ddof=1)) if s.notna().sum()>=3 else np.nan)

        out["value"] = out["symbol"].map(res)
        return out


feature = StdTradeSizeFeature()
