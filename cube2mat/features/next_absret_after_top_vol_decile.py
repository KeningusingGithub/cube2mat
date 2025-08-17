# features/next_absret_after_top_vol_decile.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class NextAbsRetAfterTopVolDecileFeature(BaseFeature):
    """
    Mean of |logret_{t+1}| conditional on volume_t in top 10% within RTH.
    NaN if <3 events.
    """
    name = "next_absret_after_top_vol_decile"
    description = "E[|logret_{t+1}| | volume_t in top decile]."
    required_full_columns = ("symbol","time","close","volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close","volume"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        for c in ("close","volume"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close","volume"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            thr=float(g["volume"].quantile(0.9))
            nxt=np.log(g["close"]).diff().abs().shift(-1)
            vals=nxt[g["volume"]>=thr].dropna()
            res[sym]=float(vals.mean()) if len(vals)>=3 else np.nan
        out["value"]=out["symbol"].map(res); return out

feature = NextAbsRetAfterTopVolDecileFeature()
