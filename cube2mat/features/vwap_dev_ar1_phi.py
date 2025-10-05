from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class VWAPDevAR1PhiFeature(BaseFeature):
    """
    AR(1) coefficient phi of deviation d_t = close - vwap in RTH:
      d_t = c + phi*d_{t-1} + e. NaN if var(d_{t-1})=0 or insufficient.
    """
    name = "vwap_dev_ar1_phi"
    description = "AR(1) phi of (close - vwap) deviation."
    required_full_columns = ("symbol","time","close","vwap")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close","vwap"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy(); 
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("close","vwap"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close","vwap"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            d=(g.sort_index()["close"]-g["vwap"]).to_numpy(float)
            if d.size<3: res[sym]=np.nan; continue
            y=d[1:]; x=d[:-1]
            X=np.column_stack([np.ones_like(x), x])
            try:
                beta = np.linalg.lstsq(X,y,rcond=None)[0]
                res[sym]=float(beta[1])
            except Exception:
                res[sym]=np.nan
        out["value"]=out["symbol"].map(res); return out

feature = VWAPDevAR1PhiFeature()
