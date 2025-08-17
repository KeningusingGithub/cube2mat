# features/next_ret_after_top_absret_decile.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class NextRetAfterTopAbsRetDecileFeature(BaseFeature):
    """
    Mean of next simple return conditional on current |logret| in top 10% within RTH.
    NaN if <3 events.
    """
    name = "next_ret_after_top_absret_decile"
    description = "E[ret_{t+1} | |logret_t| in top decile]."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(sample.symbol.unique())].copy()
        df["close"]=pd.to_numeric(df["close"],errors="coerce")
        df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            a=np.log(g["close"]).diff().abs()
            thr=float(a.quantile(0.9))
            nxt=g["close"].pct_change().shift(-1)
            vals=nxt[a>=thr].dropna()
            res[sym]=float(vals.mean()) if len(vals)>=3 else np.nan
        out["value"]=out["symbol"].map(res); return out

feature = NextRetAfterTopAbsRetDecileFeature()
