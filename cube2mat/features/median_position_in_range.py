from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class MedianPositionInRangeFeature(BaseFeature):
    """
    Median of position p_t = (close_t - L)/(H - L) in RTH, H=max(high), L=min(low).
    In [0,1]; NaN if (H-L)<=0 or no bars.
    """
    name = "median_position_in_range"
    description = "Median normalized position of close within session range."
    required_full_columns = ("symbol","time","high","low","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","high","low","close"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy(); 
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("high","low","close"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["high","low","close"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            H=float(g["high"].max()); L=float(g["low"].min())
            if H<=L: res[sym]=np.nan; continue
            p=((g["close"]-L)/(H-L)).clip(0,1)
            res[sym]=float(p.median()) if len(p)>0 else np.nan
        out["value"]=out["symbol"].map(res); return out

feature = MedianPositionInRangeFeature()
