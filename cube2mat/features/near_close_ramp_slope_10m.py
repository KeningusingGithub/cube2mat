# features/near_close_ramp_slope_10m.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class NearCloseRampSlope10mFeature(BaseFeature):
    """
    OLS slope of close on minutes since 15:50 within 15:50–15:59 window.
    NaN if <2 points or var(time)=0.
    """
    name = "near_close_ramp_slope_10m"
    description = "Tail 10-minute linear slope of close vs time."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz)
        win=df.between_time("15:50","15:59").copy()
        for c in ("close",): win[c]=pd.to_numeric(win[c],errors="coerce")
        win=win.dropna(subset=["close"])
        win=win[win.symbol.isin(sample.symbol.unique())]
        if win.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in win.groupby("symbol",sort=False):
            g=g.sort_index()
            if len(g)<2: res[sym]=np.nan; continue
            t=(g.index - g.index[0]).total_seconds()/60.0
            X=np.column_stack([np.ones(len(g)), t.to_numpy()])
            beta,_=np.linalg.lstsq(X, g["close"].to_numpy(float), rcond=None)
            res[sym]=float(beta[1])
        out["value"]=out["symbol"].map(res); return out

feature = NearCloseRampSlope10mFeature()
