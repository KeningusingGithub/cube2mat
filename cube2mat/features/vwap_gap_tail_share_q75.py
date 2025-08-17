# features/vwap_gap_tail_share_q75.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VWAPGapTailShareQ75Feature(BaseFeature):
    """
    Tail-share of |close - vwap| above its 75th percentile within 09:30–15:59.
    NaN if <3 valid bars.
    """
    name = "vwap_gap_tail_share_q75"
    description = "Share of bars where |close - vwap| exceeds its 75th percentile during RTH; NaN if <3 bars."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    q = 0.75

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
            gap = (g.sort_index()["close"] - g["vwap"]).abs().dropna()
            n = len(gap)
            if n < 3:
                res[sym] = np.nan; continue
            thr = float(gap.quantile(self.q))
            share = float((gap > thr).mean())
            res[sym] = share

        out["value"] = out["symbol"].map(res)
        return out


feature = VWAPGapTailShareQ75Feature()
