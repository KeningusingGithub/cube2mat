# features/bpv_to_rv_ratio.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from math import pi
from feature_base import BaseFeature, FeatureContext

class BPVtoRVRatioFeature(BaseFeature):
    """
    Ratio of Bipower Variation to Realized Variance using log returns in RTH:
      RV = sum r^2; BPV = (pi/2)*sum |r_t||r_{t-1}|.
    NaN if <3 returns or RV<=0.
    """
    name = "bpv_to_rv_ratio"
    description = "Bipower Variation relative to Realized Variance (log returns)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        df=df[df.symbol.isin(sample.symbol.unique())]
        df["close"]=pd.to_numeric(df["close"],errors="coerce"); df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            r=np.log(g.sort_index()["close"]).diff().replace([np.inf,-np.inf],np.nan).dropna()
            if len(r)<3: res[sym]=np.nan; continue
            RV=float((r*r).sum())
            BPV=float((pi/2.0) * np.sum(np.abs(r.values[1:])*np.abs(r.values[:-1])))
            res[sym]= (BPV/RV) if RV>0 else np.nan
        out["value"]=out["symbol"].map(res); return out

feature = BPVtoRVRatioFeature()
