# features/range_on_logvolume_beta.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RangeOnLogVolumeBetaFeature(BaseFeature):
    """
    OLS slope for (high - low) ~ 1 + log(volume) within RTH.
    Returns NaN if variance of log(volume) is 0 or insufficient data.
    """

    name = "range_on_logvolume_beta"
    description = "OLS beta of (high-low) on log(volume) in RTH."
    required_full_columns = ("symbol", "time", "high", "low", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols = ["symbol", "time", "high", "low", "volume"]
        df = self.load_full(ctx, date, cols)
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59").copy()
        for c in ("high", "low", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["high", "low", "volume"])
        df = df[df.symbol.isin(sample.symbol.unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        df["range"] = df["high"] - df["low"]
        df = df[df["range"] >= 0]

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            x = np.log(g["volume"].replace(0, np.nan))
            y = g["range"]
            xy = pd.concat([x, y], axis=1).dropna()
            if len(xy) < 3 or xy.iloc[:, 0].var() == 0:
                res[sym] = np.nan
                continue
            X = np.column_stack([np.ones(len(xy)), xy.iloc[:, 0].to_numpy()])
            beta, _ = np.linalg.lstsq(X, xy.iloc[:, 1].to_numpy(), rcond=None)
            res[sym] = float(beta[1])

        out["value"] = out["symbol"].map(res)
        return out


feature = RangeOnLogVolumeBetaFeature()
