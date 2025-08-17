# features/mean_close_vwap_rel_diff.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class MeanCloseVWAPRelDiffFeature(BaseFeature):
    """
    Mean relative premium of close vs vwap: mean((close - vwap) / vwap) in RTH.
    """

    name = "mean_close_vwap_rel_diff"
    description = "Average (close - vwap)/vwap over RTH bars."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "close", "vwap"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59").copy()
        for c in ("close", "vwap"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "vwap"])
        df = df[df.symbol.isin(sample.symbol.unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        res = df.groupby("symbol").apply(
            lambda g: float(((g["close"] - g["vwap"]) / g["vwap"]).mean()) if len(g) > 0 else np.nan
        )
        out["value"] = out["symbol"].map(res)
        return out


feature = MeanCloseVWAPRelDiffFeature()
