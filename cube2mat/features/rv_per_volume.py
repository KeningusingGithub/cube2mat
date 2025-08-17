from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class RVPerVolumeFeature(BaseFeature):
    """
    Realized variance per unit volume in RTH:
      r=diff(log close); RV=sum r^2; value = RV / sum(volume). NaN if RV<=0 or sum(volume)<=0.
    """
    name = "rv_per_volume"
    description = "Realized variance divided by total volume (RTH)."
    required_full_columns = ("symbol","time","close","volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close","volume"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy();
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        for c in ("close","volume"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close","volume"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            r=np.log(g["close"]).diff().replace([np.inf,-np.inf],np.nan).dropna()
            if len(r)<3: res[sym]=np.nan; continue
            RV=float((r*r).sum())
            vol=float(g["volume"].sum())
            res[sym]= (RV/vol) if (RV>0 and vol>0) else np.nan
        out["value"]=out["symbol"].map(res); return out

feature = RVPerVolumeFeature()
