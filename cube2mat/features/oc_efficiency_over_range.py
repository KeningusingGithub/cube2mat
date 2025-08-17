from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class OCEfficiencyOverRangeFeature(BaseFeature):
    """
    Directional efficiency of session:
      eff = |last_close - first_open| / (H - L), H=max(high), L=min(low) in RTH.
    NaN if (H-L)<=0 or anchors missing.
    """
    name = "oc_efficiency_over_range"
    description = "Absolute OC move normalized by session range (H-L)."
    required_full_columns = ("symbol","time","open","high","low","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","open","high","low","close"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy(); 
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df = self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("open","high","low","close"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["open","high","low","close"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            H=float(g["high"].max()); L=float(g["low"].min())
            if H<=L: res[sym]=np.nan; continue
            o0=float(g["open"].iloc[0]); cL=float(g["close"].iloc[-1])
            res[sym]=abs(cL-o0)/(H-L)
        out["value"]=out["symbol"].map(res); return out

feature = OCEfficiencyOverRangeFeature()
