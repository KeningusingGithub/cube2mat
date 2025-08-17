# features/impact_slope_price_cumn.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class ImpactSlopePriceCumNFeature(BaseFeature):
    """
    OLS slope of (close - first_open) on cumulative trade count within RTH.
    NaN if insufficient or var(cum_n)=0.
    """
    name = "impact_slope_price_cumn"
    description = "OLS beta of (close - first_open) on cumulative n (RTH)."
    required_full_columns = ("symbol","time","open","close","n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","open","close","n"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("open","close","n"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close","n"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            if g["open"].dropna().empty or len(g)<2: res[sym]=np.nan; continue
            anchor=float(g["open"].dropna().iloc[0])
            y=(g["close"]-anchor)
            x=g["n"].cumsum()
            xy=pd.concat([x,y],axis=1).dropna()
            if len(xy)<3 or xy.iloc[:,0].var()==0: res[sym]=np.nan; continue
            X=np.column_stack([np.ones(len(xy)), xy.iloc[:,0].to_numpy()])
            beta,_=np.linalg.lstsq(X, xy.iloc[:,1].to_numpy(), rcond=None)
            res[sym]=float(beta[1])
        out["value"]=out["symbol"].map(res); return out

feature = ImpactSlopePriceCumNFeature()
