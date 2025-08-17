# features/mean_abs_vwap_gap.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class MeanAbsVWAPGapFeature(BaseFeature):
    """
    Mean absolute gap between close and vwap within 09:30–15:59.
    NaN if <1 valid gap.
    """
    name = "mean_abs_vwap_gap"
    description = "Mean of |close - vwap| during RTH; NaN if no valid bars."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, list(self.required_pv_columns))
        if df_full is None or sample is None: return None
        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty: out["value"] = pd.NA; return out

        df = self.ensure_et_index(df_full, "time", ctx.tz).between_time("09:30","15:59")
        if df.empty: out["value"] = pd.NA; return out

        for col in ("close","vwap"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close","vwap"])
        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
        if df.empty: out["value"] = pd.NA; return out

        gap = (df["close"] - df["vwap"]).abs()
        res = df.assign(gap=gap).groupby("symbol")["gap"].agg(lambda s: float(s.mean()) if s.notna().sum()>=1 else np.nan)

        out["value"] = out["symbol"].map(res)
        return out


feature = MeanAbsVWAPGapFeature()
