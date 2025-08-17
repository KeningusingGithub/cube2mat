from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class RVDiffCloseVWAPFeature(BaseFeature):
    """
    相对 VWAP 溢价（close - vwap）的已实现“变差”（一阶差分的平方和）：
      x_t = close_t - vwap_t；RV_diff = sum( (x_t - x_{t-1})^2 )。
    需成对有效 close/vwap；有效差分<1 则 NaN。
    """
    name = "rv_close_vwap_diff"
    description = "Realized variance of (close - vwap): sum of squared first differences."
    required_full_columns = ("symbol","time","close","vwap")
    required_pv_columns   = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full=self.load_full(ctx,date,list(self.required_full_columns))
        sample=self.load_pv(ctx,date,list(self.required_pv_columns))
        if full is None or sample is None: return None

        out=sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"]=pd.NA; return out

        df=self.ensure_et_index(full,"time",ctx.tz).between_time("09:30","15:59").sort_index()
        df=df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty:
            out["value"]=pd.NA; return out

        for c in ("close","vwap"):
            df[c]=pd.to_numeric(df[c], errors="coerce")
        df=df.dropna(subset=["close","vwap"])
        if df.empty:
            out["value"]=pd.NA; return out

        df["x"]=df["close"] - df["vwap"]
        df["dx"]=df.groupby("symbol", sort=False)["x"].diff()
        df["dx"]=df["dx"].replace([np.inf,-np.inf], np.nan)
        df=df.dropna(subset=["dx"])
        if df.empty:
            out["value"]=pd.NA; return out

        value = df.groupby("symbol")["dx"].apply(lambda s: float((s*s).sum()) if len(s)>=1 else np.nan)
        out["value"]=out["symbol"].map(value)
        return out

feature = RVDiffCloseVWAPFeature()
