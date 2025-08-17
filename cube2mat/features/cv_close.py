# features/cv_close.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class CVCloseFeature(BaseFeature):
    """
    09:30–15:59 内，CV(close) = std(close)/mean(close)；mean<=0 或样本<2 则 NaN。
    """

    name = "cv_close"
    description = "Coefficient of variation of close: std/mean within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

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
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        g = df.groupby("symbol")["close"]
        stats = g.agg(n="count", mean="mean", std=lambda s: s.std(ddof=1))
        cv = (stats["std"] / stats["mean"]).where(
            (stats["n"] >= 2) & (stats["mean"] > 0)
        )
        out["value"] = out["symbol"].map(cv)
        return out


feature = CVCloseFeature()

