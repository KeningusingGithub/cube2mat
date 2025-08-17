# features/cv_trade_size.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class CVTradeSizeFeature(BaseFeature):
    """
    Coefficient of variation for per-bar trade size (volume/n):
        value = std(tsize)/mean(tsize), unbiased std; NaN if <3 bars or mean<=0.
    """
    name = "cv_trade_size"
    description = "Coefficient of variation of per-bar trade size (volume/n) within 09:30–15:59."
    required_full_columns = ("symbol", "time", "volume", "n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, list(self.required_pv_columns))
        if df_full is None or sample is None: return None
        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty:
            out["value"] = pd.NA; return out

        df = self.ensure_et_index(df_full, "time", ctx.tz).between_time("09:30","15:59")
        if df.empty: out["value"] = pd.NA; return out

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))].copy()
        for col in ("volume","n"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["volume","n"])
        if df.empty: out["value"] = pd.NA; return out

        df = df[df["n"] > 0]
        if df.empty: out["value"] = pd.NA; return out

        df["tsize"] = df["volume"]/df["n"]

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            s = g["tsize"].dropna()
            if len(s) < 3: res[sym] = np.nan; continue
            mu = float(s.mean())
            sd = float(s.std(ddof=1))
            res[sym] = sd/mu if (np.isfinite(mu) and mu>0) else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = CVTradeSizeFeature()
