# features/vwap_cross_count.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VWAPCrossCountFeature(BaseFeature):
    """
    Count sign changes of (close - vwap) within 09:30–15:59, ignoring zeros.
    Crossing counted when consecutive non-zero signs differ. NaN if <2 non-zero points.
    """
    name = "vwap_cross_count"
    description = "Count of sign flips of (close - vwap) during RTH; ignore zeros."
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

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            d = (g.sort_index()["close"] - g["vwap"]).to_numpy()
            s = np.sign(d)
            s = s[s != 0]
            if s.size < 2:
                res[sym] = np.nan; continue
            flips = np.sum(s[1:] != s[:-1])
            res[sym] = float(flips)
        out["value"] = out["symbol"].map(res)
        return out


feature = VWAPCrossCountFeature()
