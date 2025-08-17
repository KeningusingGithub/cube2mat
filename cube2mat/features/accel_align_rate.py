from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class AccelAlignRateFeature(BaseFeature):
    """
    Share of bars where acceleration and velocity are aligned:
      d1_t = Δclose_t, d2_t = Δ²close_t; align on d2 index (skip first two points);
      value = mean( 1[(d1_{t})*d2_t > 0] ), excluding zeros. NaN if <1 valid pair.
    """
    name = "accel_align_rate"
    description = "Fraction of time with curvature pushing in the velocity direction."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy(); 
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(sample.symbol.unique())]
        df["close"]=pd.to_numeric(df["close"],errors="coerce"); df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            c=g.sort_index()["close"].to_numpy(float)
            if c.size<4: res[sym]=np.nan; continue
            d1=np.diff(c); d2=np.diff(d1)
            x=d1[1:]; y=d2  # align
            mask=(x!=0) & (y!=0)
            if mask.sum()<1: res[sym]=np.nan; continue
            val=float(((x[mask]*y[mask])>0).mean())
            res[sym]=val
        out["value"]=out["symbol"].map(res); return out

feature = AccelAlignRateFeature()
