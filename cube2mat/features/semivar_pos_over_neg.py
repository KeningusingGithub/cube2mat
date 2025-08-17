# features/semivar_pos_over_neg.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class SemivarPosOverNegFeature(BaseFeature):
    """
    09:30–15:59 内，上/下行半方差比率；若下行半方差<=0 则 NaN。
    """

    name = "semivar_pos_over_neg"
    description = "Ratio of positive to negative semivariance of log returns."
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
            pos = float(np.sum((r[r > 0]) ** 2))
            neg = float(np.sum((r[r < 0]) ** 2))
            return pos / neg if np.isfinite(pos) and np.isfinite(neg) and neg > 0 else np.nan

        value = df.groupby("symbol")["r"].apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = SemivarPosOverNegFeature()

