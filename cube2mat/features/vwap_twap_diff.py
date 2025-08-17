# features/vwap_twap_diff.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VWAPTWAPDiffFeature(BaseFeature):
    """
    Session VWAP minus TWAP within 09:30–15:59:
      - TWAP = mean(close)
      - VWAP = sum(close*volume)/sum(volume)
      value = VWAP - TWAP; NaN if sum(volume)<=0 or <3 ticks.
    """
    name = "vwap_twap_diff"
    description = "VWAP - TWAP (mean(close)) within 09:30–15:59; NaN if insufficient."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, list(self.required_pv_columns))
        if df_full is None or sample is None: return None
        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty: out["value"] = pd.NA; return out

        df = self.ensure_et_index(df_full, "time", ctx.tz).between_time("09:30","15:59")
        if df.empty: out["value"] = pd.NA; return out

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))].copy()
        for col in ("close","volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close","volume"])
        if df.empty: out["value"] = pd.NA; return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            if len(g) < 3: res[sym] = np.nan; continue
            twap = float(g["close"].mean())
            vsum = float(g["volume"].sum())
            if not np.isfinite(vsum) or vsum <= 0:
                res[sym] = np.nan; continue
            vwap = float((g["close"] * g["volume"]).sum() / vsum)
            res[sym] = vwap - twap
        out["value"] = out["symbol"].map(res)
        return out


feature = VWAPTWAPDiffFeature()
