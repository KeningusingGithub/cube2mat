from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class CLVSessionFeature(BaseFeature):
    """
    Close Location Value over session range:
      CLV = ((last_close - L) - (H - last_close)) / (H - L), H=max(high), L=min(low)
    NaN if (H-L)<=0 or no bars.
    """
    name = "clv_session"
    description = "Session Close Location Value in RTH using session H/L and last close."
    required_full_columns = ("symbol","time","high","low","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","high","low","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy()

        if df.empty or sample.empty:
            out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("high","low","close"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["high","low","close"])
        df=df[df["symbol"].isin(sample["symbol"].unique())]

        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            H=float(g["high"].max()); L=float(g["low"].min())
            if not np.isfinite(H) or not np.isfinite(L) or (H-L)<=0 or len(g)<1:
                res[sym]=np.nan; continue
            c=float(g["close"].iloc[-1])
            res[sym]= ((c - L) - (H - c)) / (H - L)
        out["value"]=out["symbol"].map(res); return out


feature = CLVSessionFeature()
