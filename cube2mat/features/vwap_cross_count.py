# features/vwap_cross_count.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VWAPCrossCountFeature(BaseFeature):
    """
    Count of strict sign flips between close and vwap within RTH.
    Crossing occurs when sign(diff_t) * sign(diff_{t-1}) < 0, ignoring zeros.
    """

    name = "vwap_cross_count"
    description = "Number of strict sign flips of (close - vwap) in RTH."
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
            diff = (g.sort_index()["close"] - g.sort_index()["vwap"]).to_numpy(dtype=float)
            if diff.size < 2:
                res[sym] = np.nan
                continue
            s = np.sign(diff)
            mask = s != 0
            s = s[mask]
            if s.size < 2:
                res[sym] = np.nan
                continue
            crosses = int(np.sum(s[1:] * s[:-1] < 0))
            res[sym] = float(crosses)

        out["value"] = out["symbol"].map(res)
        return out


feature = VWAPCrossCountFeature()
