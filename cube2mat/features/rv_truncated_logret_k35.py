# features/rv_truncated_logret_k35.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RVTruncatedLogretK35Feature(BaseFeature):
    """
    Realized variance of log returns after truncation by robust MAD threshold.
    Steps:
      1) 09:30–15:59, per symbol logret r = diff(log(close))
      2) Robust scale via MAD: sigma_mad = 1.4826 * median(|r - median(r)|)
      3) Keep |r| <= k*sigma_mad (k=3.5); if sigma_mad==0 fallback to keep all
      4) RV = sum(r^2) over kept returns; NaN if <3 returns after filtering.
    """
    name = "rv_truncated_logret_k35"
    description = "Realized variance of log returns after truncation at 3.5×MAD within 09:30–15:59; NaN if <3 returns."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)
    k = 3.5

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        if df_full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df_full, time_col="time", tz=ctx.tz).between_time("09:30", "15:59")
        if df.empty:
            out["value"] = pd.NA
            return out

        syms = set(sample["symbol"].unique())
        df = df[df["symbol"].isin(syms)]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        result = {}
        for sym, g in df.groupby("symbol", sort=False):
            s = np.log(g.sort_index()["close"]).diff()
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 3:
                result[sym] = np.nan
                continue
            med = s.median()
            mad = (s - med).abs().median()
            sigma_mad = 1.4826 * mad
            if sigma_mad > 0:
                thr = self.k * sigma_mad
                s = s[(s.abs() <= thr)]
            if len(s) < 3:
                result[sym] = np.nan
            else:
                result[sym] = float((s * s).sum())

        out["value"] = out["symbol"].map(result)
        return out


feature = RVTruncatedLogretK35Feature()
