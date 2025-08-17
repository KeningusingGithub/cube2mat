# features/vwap_dev_bps_time_share_10bp.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class VWAPDevBPSTimeShare10bpFeature(BaseFeature):
    """
    Fraction of bars with |close - vwap| / vwap >= 10bp (0.001) within RTH.
    """
    name = "vwap_dev_bps_time_share_10bp"
    description = "Time share with |close-vwap|/vwap >= 10bp."
    required_full_columns = ("symbol","time","close","vwap")
    required_pv_columns = ("symbol",)
    thr = 1e-3  # 10 bps

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close","vwap"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("close","vwap"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close","vwap"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out
        thr=float(self.thr)
        res=df.assign(dev=((df["close"]-df["vwap"]).abs()/df["vwap"])).groupby("symbol")["dev"].apply(
            lambda s: float((s>=thr).mean()) if len(s)>0 else np.nan
        )
        out["value"]=out["symbol"].map(res); return out

feature = VWAPDevBPSTimeShare10bpFeature()
