# features/rv_per_min_logret.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RVPerMinLogRetFeature(BaseFeature):
    """
    09:30–15:59 内，RV 每分钟强度： sum(diff(log(close))^2) / TOTAL_MIN；close>0。
    """

    name = "rv_per_min_logret"
    description = "RV per-minute intensity: RV divided by session total minutes (389)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)
    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

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

        rv = df.groupby("symbol")["r"].apply(lambda s: (s * s).sum())
        value = rv.apply(lambda x: float(x / self.TOTAL_MIN) if pd.notna(x) else np.nan)
        out["value"] = out["symbol"].map(value)
        return out


feature = RVPerMinLogRetFeature()

