# features/volume_skew.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class VolumeSkewFeature(BaseFeature):
    """
    Sample skewness of per-bar volume within RTH:
      g1 = m3 / s^3 (Fisher). NaN if <3 bars or s==0.
    """
    name = "volume_skew"
    description = "Skewness of volume distribution across RTH bars."
    required_full_columns = ("symbol","time","volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","volume"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(sample.symbol.unique())]
        df["volume"]=pd.to_numeric(df["volume"],errors="coerce"); df=df.dropna(subset=["volume"])
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            v=g.sort_index()["volume"].to_numpy(float)
            n=v.size
            if n<3: res[sym]=np.nan; continue
            mu=v.mean(); s=v.std(ddof=1)
            if s<=0: res[sym]=np.nan; continue
            m3=float(np.mean((v-mu)**3))
            res[sym]=float(m3/(s**3))
        out["value"]=out["symbol"].map(res); return out

feature = VolumeSkewFeature()
