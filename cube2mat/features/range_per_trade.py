from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class RangePerTradeFeature(BaseFeature):
    """
    Session range per trade: (H-L)/sum(n) within RTH. NaN if H<=L or sum(n)<=0.
    """
    name = "range_per_trade"
    description = "Price range divided by total trade count in RTH."
    required_full_columns = ("symbol","time","high","low","n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","high","low","n"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy(); 
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        for c in ("high","low","n"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["high","low","n"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            H=float(g["high"].max()); L=float(g["low"].min())
            rng=H-L; tn=float(g["n"].sum())
            res[sym]=(rng/tn) if (rng>0 and tn>0) else np.nan
        out["value"]=out["symbol"].map(res); return out

feature = RangePerTradeFeature()
