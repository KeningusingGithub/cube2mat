# features/midday_lull_rv_ratio.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class MiddayLullRVRatioFeature(BaseFeature):
    """
    RV density ratio at midday (12:00–12:59):
      ratio = (sum r^2_mid / sum r^2_total) / (count_ret_mid / count_ret_total).
    NaN if RV_total<=0 or insufficient returns.
    """
    name = "midday_lull_rv_ratio"
    description = "RV density at midday vs session average."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz)
        rth=df.between_time("09:30","15:59")
        rth=rth[rth.symbol.isin(sample.symbol.unique())].copy()
        rth["close"]=pd.to_numeric(rth["close"],errors="coerce"); rth=rth.dropna(subset=["close"])
        if rth.empty: out["value"]=pd.NA; return out

        mid=rth.between_time("12:00","12:59")
        res={}
        for sym,g in rth.groupby("symbol",sort=False):
            g=g.sort_index()
            r=np.log(g["close"]).diff().replace([np.inf,-np.inf],np.nan).iloc[1:]
            total=float((r**2).sum()); cnt_total=int(r.shape[0])
            m=mid[mid.symbol==sym].sort_index()
            if m.empty or cnt_total<3 or total<=0:
                res[sym]=np.nan; continue
            rv_mid=float((r[r.index.isin(m.index)]**2).sum())
            cnt_mid=int(r.index.isin(m.index).sum())
            if cnt_mid==0: res[sym]=np.nan; continue
            ratio=(rv_mid/total)/ (cnt_mid/max(1,cnt_total))
            res[sym]=float(ratio)
        out["value"]=out["symbol"].map(res); return out

feature = MiddayLullRVRatioFeature()
