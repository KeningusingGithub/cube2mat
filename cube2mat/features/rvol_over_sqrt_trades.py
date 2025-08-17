# features/rvol_over_sqrt_trades.py
from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class RVolOverSqrtTradesFeature(BaseFeature):
    """
    Invariance proxy: sqrt( sum r^2 ) / sqrt( sum n ), r=log returns in RTH.
    NaN if RV<=0 or sum(n)<=0 or <3 returns.
    """
    name = "rvol_over_sqrt_trades"
    description = "sqrt(RV)/sqrt(total trades) invariance proxy (RTH)."
    required_full_columns = ("symbol","time","close","n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close","n"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        for c in ("close","n"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close","n"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            r=np.log(g["close"]).diff().replace([np.inf,-np.inf],np.nan).dropna()
            if len(r)<3: res[sym]=np.nan; continue
            RV=float((r*r).sum()); TN=float(g["n"].sum())
            if RV<=0 or TN<=0: res[sym]=np.nan; continue
            res[sym]=float(np.sqrt(RV)/np.sqrt(TN))
        out["value"]=out["symbol"].map(res); return out

feature = RVolOverSqrtTradesFeature()
