# features/entropy_sign_ret.py
from __future__ import annotations
import datetime as dt
import math
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class EntropySignRetFeature(BaseFeature):
    """
    Normalized Shannon entropy of simple-return signs in RTH:
    states = {-1, 0, +1}; H = -sum p_i log p_i / log(k), where k is number of states with
    non-zero probability. Returns NaN if fewer than one return or only one state present.
    """

    name = "entropy_sign_ret"
    description = "Normalized entropy (0..1) of simple-return sign distribution."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59")
        df = df[df.symbol.isin(sample.symbol.unique())].copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            r = (
                g.sort_index()["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            )
            if len(r) < 1:
                res[sym] = np.nan
                continue
            s = np.sign(r.values)
            vals, counts = np.unique(s, return_counts=True)
            p = counts / counts.sum()
            p = p[p > 0]
            k = p.size
            if k < 2:
                res[sym] = np.nan
                continue
            H = -np.sum(p * np.log(p)) / math.log(k)
            res[sym] = float(np.clip(H, 0.0, 1.0))

        out["value"] = out["symbol"].map(res)
        return out


feature = EntropySignRetFeature()
