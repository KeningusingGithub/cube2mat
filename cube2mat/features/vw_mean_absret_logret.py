from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class VWMeanAbsRetLogretFeature(BaseFeature):
    """
    Volume-weighted mean of |log returns| in RTH:
      r = diff(log close); use weights w_t = volume_t aligned to r (end time).
      value = sum w*|r| / sum w. NaN if sum w<=0 or <1 return.
    """
    name = "vw_mean_absret_logret"
    description = "Volume-weighted mean absolute log return."
    required_full_columns = ("symbol","time","close","volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close","volume"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy();
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("close","volume"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close","volume"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            r=np.log(g["close"]).diff().abs()
            w=g["volume"].iloc[1:]
            xy=pd.concat([r.iloc[1:], w],axis=1).dropna()
            if xy.empty or xy.iloc[:,1].sum()<=0: res[sym]=np.nan; continue
            val=float((xy.iloc[:,0]*xy.iloc[:,1]).sum()/xy.iloc[:,1].sum())
            res[sym]=val
        out["value"]=out["symbol"].map(res); return out

feature = VWMeanAbsRetLogretFeature()
