# features/ret_next_on_vwapdev_beta.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RetNextOnVWAPDevBetaFeature(BaseFeature):
    """
    OLS slope of next simple return on normalized deviation d_t = (close - vwap) / vwap.
    y = ret_{t+1}; x = d_t within RTH.
    Returns NaN if insufficient data or zero variance.
    """

    name = "ret_next_on_vwapdev_beta"
    description = "Beta of next ret on (close-vwap)/vwap deviation (RTH)."
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

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            dv = (g["close"] - g["vwap"]) / g["vwap"]
            y = g["close"].pct_change().shift(-1)
            xy = pd.concat([dv, y], axis=1).dropna()
            if len(xy) < 3 or xy.iloc[:, 0].var() == 0:
                res[sym] = np.nan
                continue
            X = np.column_stack([np.ones(len(xy)), xy.iloc[:, 0].to_numpy()])
            beta = np.linalg.lstsq(X, xy.iloc[:, 1].to_numpy(), rcond=None)[0]
            res[sym] = float(beta[1])

        out["value"] = out["symbol"].map(res)
        return out


feature = RetNextOnVWAPDevBetaFeature()
