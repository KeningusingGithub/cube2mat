# features/vol_of_vol_absret.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VolOfVolAbsRetFeature(BaseFeature):
    """
    Vol-of-Vol proxy using absolute log returns:
      1) r = diff(log(close)), a = |r|
      2) compute std of Δa = a - a.shift(1), unbiased (ddof=1)
      NaN if <3 valid Δa.
    """
    name = "vol_of_vol_absret"
    description = "Std of first differences of |log returns| within 09:30–15:59; NaN if <3 valid diffs."
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
            r = np.log(g.sort_index()["close"]).diff().replace([np.inf, -np.inf], np.nan).dropna()
            a = r.abs()
            da = a.diff().dropna()
            res[sym] = float(da.std(ddof=1)) if len(da) >= 3 else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = VolOfVolAbsRetFeature()
