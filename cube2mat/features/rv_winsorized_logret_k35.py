# features/rv_winsorized_logret_k35.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RVWinsorizedLogretK35Feature(BaseFeature):
    """
    Realized variance of log returns with winsorization at 3.5×MAD.
    Steps:
      1) r = diff(log(close)) within 09:30–15:59
      2) sigma_mad = 1.4826 * MAD(r)
      3) r_w = clip(r, -k*sigma_mad, +k*sigma_mad) with k=3.5 (if sigma_mad>0)
      4) RV = sum(r_w^2); NaN if <3 returns.
    """
    name = "rv_winsorized_logret_k35"
    description = "Winsorized realized variance of log returns at 3.5×MAD within 09:30–15:59; NaN if <3 returns."
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

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            s = np.log(g.sort_index()["close"]).diff()
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 3:
                res[sym] = np.nan
                continue
            med = s.median()
            mad = (s - med).abs().median()
            sigma_mad = 1.4826 * mad
            if sigma_mad > 0:
                thr = self.k * sigma_mad
                s = s.clip(lower=-thr, upper=thr)
            if len(s) < 3:
                res[sym] = np.nan
            else:
                res[sym] = float((s * s).sum())
        out["value"] = out["symbol"].map(res)
        return out


feature = RVWinsorizedLogretK35Feature()
