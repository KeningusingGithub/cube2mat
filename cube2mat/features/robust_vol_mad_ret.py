# features/robust_vol_mad_ret.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RobustVolMADRetFeature(BaseFeature):
    """
    09:30–15:59 内，基于 log 收益的稳健波动率：
      sigma ≈ 1.4826 * median(|r - median(r)|)；close>0。
    """

    name = "robust_vol_mad_ret"
    description = "Robust volatility via MAD on log returns."
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

        def per_symbol(s: pd.Series) -> float:
            r = s.values
            if len(r) < 2:
                return np.nan
            med = np.median(r)
            mad = np.median(np.abs(r - med))
            return float(1.4826 * mad) if np.isfinite(mad) else np.nan

        value = df.groupby("symbol")["r"].apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = RobustVolMADRetFeature()

