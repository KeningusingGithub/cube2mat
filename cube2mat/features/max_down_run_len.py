# features/max_down_run_len.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class MaxDownRunLenFeature(BaseFeature):
    """
    最长连跌长度（ret<0 的连续段最大长度）。
    """

    name = "max_down_run_len"
    description = "Maximum consecutive length of negative-return run."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _max_run(mask: pd.Series) -> float:
        if mask.empty:
            return np.nan
        run_id = (mask != mask.shift()).cumsum()
        lengths = mask.groupby(run_id).sum()
        if lengths.empty:
            return np.nan
        return float(lengths.max()) if lengths.max() > 0 else np.nan

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59").sort_index()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(sample["symbol"].unique())]
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret"])

        value = df.groupby("symbol").apply(lambda g: self._max_run(g["ret"] < 0))
        out["value"] = out["symbol"].map(value)
        return out


feature = MaxDownRunLenFeature()
