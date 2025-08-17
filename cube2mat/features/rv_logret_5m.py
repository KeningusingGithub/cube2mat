# features/rv_logret_5m.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RVLogret5mFeature(BaseFeature):
    """
    Downsample close to 5-minute bars within 09:30–15:59 (last price per bin),
    then compute RV using log returns on the resampled series.
    """
    name = "rv_logret_5m"
    description = "Realized variance of log returns on 5-minute resampled close within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

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

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            s = g.sort_index()["close"].resample("5T").last().dropna()
            if len(s) < 3:
                res[sym] = np.nan
                continue
            r = np.log(s).diff().replace([np.inf, -np.inf], np.nan).dropna()
            res[sym] = float((r * r).sum()) if len(r) >= 2 else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = RVLogret5mFeature()
