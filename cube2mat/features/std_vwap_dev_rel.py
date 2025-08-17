# features/std_vwap_dev_rel.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class StdVWAPDevRelFeature(BaseFeature):
    """
    Standard deviation of relative deviation (close - vwap)/vwap within RTH.
    NaN if <2 bars or any vwap<=0.
    """
    name = "std_vwap_dev_rel"
    description = "Std of (close-vwap)/vwap across RTH bars."
    required_full_columns = ("symbol","time","close","vwap")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close","vwap"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        for c in ("close","vwap"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close","vwap"])
        df=df[df["vwap"]>0]
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out
        res=df.assign(z=(df["close"]-df["vwap"])/df["vwap"]).groupby("symbol")["z"].apply(
            lambda s: float(s.std(ddof=1)) if len(s)>=2 else np.nan
        )
        out["value"]=out["symbol"].map(res); return out

feature = StdVWAPDevRelFeature()
