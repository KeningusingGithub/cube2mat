# features/kyle_lambda_lead1_beta.py
from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class KyleLambdaLead1BetaFeature(BaseFeature):
    """
    OLS slope (beta) of next simple return on signed volume proxy:
      x_t = sign(ret_t) * volume_t, y_t = ret_{t+1}.
    Reduces endogeneity vs contemporaneous spec. NaN if insufficient.
    """
    name = "kyle_lambda_lead1_beta"
    description = "Beta of ret_{t+1} on sign(ret_t)*volume_t (Kyle λ proxy)."
    required_full_columns = ("symbol","time","close","volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close","volume"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
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
            r=g["close"].pct_change()
            x=np.sign(r)*g["volume"]
            y=r.shift(-1)
            xy=pd.concat([x,y],axis=1).dropna()
            if len(xy)<3 or xy.iloc[:,0].var()==0: res[sym]=np.nan; continue
            X=np.column_stack([np.ones(len(xy)), xy.iloc[:,0].to_numpy()])
            beta,_=np.linalg.lstsq(X, xy.iloc[:,1].to_numpy(), rcond=None)
            res[sym]=float(beta[1])
        out["value"]=out["symbol"].map(res); return out

feature = KyleLambdaLead1BetaFeature()
