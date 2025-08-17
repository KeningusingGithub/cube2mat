# features/n_gini.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

def _gini(x: np.ndarray) -> float:
    x=np.asarray(x,float); x=x[np.isfinite(x)&(x>=0)]
    n=x.size; s=x.sum()
    if n<2 or s<=0: return np.nan
    xs=np.sort(x); i=np.arange(1,n+1)
    g=1.0 - 2.0*np.sum((n - i + 0.5)*xs)/(n*s)
    return float(np.clip(g,0.0,1.0))

class NGiniFeature(BaseFeature):
    """
    Gini index of trade count distribution across RTH bars.
    """
    name = "n_gini"
    description = "Gini of per-bar trade counts (n) in RTH."
    required_full_columns = ("symbol","time","n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","n"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(sample.symbol.unique())]
        df["n"]=pd.to_numeric(df["n"],errors="coerce"); df=df.dropna(subset=["n"])
        if df.empty: out["value"]=pd.NA; return out

        res=df.groupby("symbol")["n"].apply(lambda s: _gini(s.values))
        out["value"]=out["symbol"].map(res); return out

feature = NGiniFeature()
