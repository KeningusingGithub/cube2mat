# features/theil_sen_slope_close_time.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class TheilSenSlopeCloseTimeFeature(BaseFeature):
    """
    Theil–Sen robust slope of close on minutes since 09:30 within RTH.
    Median of pairwise slopes (y_j - y_i)/(t_j - t_i), i<j. NaN if <2 points.
    """
    name = "theil_sen_slope_close_time"
    description = "Robust trend slope (Theil–Sen) of close vs time (minutes)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def _theil_sen(self, t: np.ndarray, y: np.ndarray) -> float:
        n=y.size
        if n<2: return np.nan
        slopes=[]
        for i in range(n-1):
            dt=t[i+1:]-t[i]
            dy=y[i+1:]-y[i]
            m=dy[dt!=0]/dt[dt!=0]
            if m.size>0: slopes.append(m)
        if not slopes: return np.nan
        allm=np.concatenate(slopes)
        return float(np.median(allm))

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
            g=g.sort_index()
            t=((g.index - g.index[0]).total_seconds()/60.0).to_numpy(float)
            y=g["close"].to_numpy(float)
            res[sym]=self._theil_sen(t,y)
        out["value"]=out["symbol"].map(res); return out

feature = TheilSenSlopeCloseTimeFeature()
