# features/duration_since_last_vwap_cross_frac.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

TOT_MIN = 389.0

class DurationSinceLastVWAPCrossFracFeature(BaseFeature):
    """
    Fraction of session elapsed since the last strict cross of (close-vwap) sign.
    If no cross in RTH, returns NaN.
    """
    name = "duration_since_last_vwap_cross_frac"
    description = "Session fraction since last VWAP crossing (stickiness)."
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
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            d=(g["close"]-g["vwap"]).to_numpy(float)
            if d.size<2: res[sym]=np.nan; continue
            s=np.sign(d)
            cross_idx=np.where(s[1:]*s[:-1] < 0)[0]
            if cross_idx.size==0:
                res[sym]=np.nan; continue
            last_cross_time=g.index[cross_idx[-1]+1]
            frac=float((g.index[-1]-last_cross_time).total_seconds()/60.0)/TOT_MIN
            res[sym]=float(np.clip(frac,0.0,1.0))
        out["value"]=out["symbol"].map(res); return out

feature = DurationSinceLastVWAPCrossFracFeature()
