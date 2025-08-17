# features/signed_volume_imbalance.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class SignedVolumeImbalanceFeature(BaseFeature):
    """
    Net signed volume fraction using tick-rule proxy:
      value = sum(sign(simple ret)*volume) / sum(volume) in RTH.
    NaN if total volume<=0 or <1 return.
    """
    name = "signed_volume_imbalance"
    description = "Signed volume share ∈ [-1,1] based on ret sign proxy."
    required_full_columns = ("symbol","time","close","volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close","volume"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("close","volume"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close","volume"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            r=g["close"].pct_change().replace([np.inf,-np.inf],np.nan)
            if r.dropna().empty: res[sym]=np.nan; continue
            s=np.sign(r)
            s=s.fillna(0.0)
            num=float((s*g["volume"]).sum()); den=float(g["volume"].sum())
            res[sym]= (num/den) if den>0 else np.nan
        out["value"]=out["symbol"].map(res); return out

feature = SignedVolumeImbalanceFeature()
