# features/vwap_zscore_close_end.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VWAPZScoreCloseEndFeature(BaseFeature):
    """
    Z-score of last close vs session VWAP:
      z = (last_close - VWAP_session) / std(close - vwap)
    VWAP_session = sum(close*volume)/sum(volume) (not per-bar vwap).
    NaN if std==0 or <3 bars or sum(volume)<=0.
    """
    name = "vwap_zscore_close_end"
    description = "Z-score of last close vs session VWAP using std(close - vwap) as scale; RTH only."
    required_full_columns = ("symbol", "time", "close", "vwap", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, list(self.required_pv_columns))
        if df_full is None or sample is None: return None
        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty: out["value"] = pd.NA; return out

        df = self.ensure_et_index(df_full, "time", ctx.tz).between_time("09:30","15:59")
        if df.empty: out["value"] = pd.NA; return out

        for col in ("close","vwap","volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close","vwap","volume"])
        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
        if df.empty: out["value"] = pd.NA; return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            if len(g) < 3:
                res[sym] = np.nan; continue
            last_close = float(g["close"].iloc[-1])
            wsum = float(g["volume"].sum())
            if not np.isfinite(wsum) or wsum <= 0:
                res[sym] = np.nan; continue
            vwap_session = float((g["close"] * g["volume"]).sum() / wsum)
            diff = (g["close"] - g["vwap"]).dropna()
            sd = float(diff.std(ddof=1)) if len(diff) >= 2 else np.nan
            if not np.isfinite(sd) or sd == 0:
                res[sym] = np.nan
            else:
                res[sym] = (last_close - vwap_session) / sd
        out["value"] = out["symbol"].map(res)
        return out


feature = VWAPZScoreCloseEndFeature()
