# features/dc_count_bps_10.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class DCCountBps10Feature(BaseFeature):
    """
    Directional-change count with threshold theta=10bp (0.001) on close.
    Event when price reverses by >=theta from the last swing extreme. NaN if <2 closes.
    """
    name = "dc_count_bps_10"
    description = "Directional-change event count at 10bp threshold."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)
    theta = 1e-3  # 10 bps

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(sample.symbol.unique())]
        df["close"]=pd.to_numeric(df["close"],errors="coerce")
        df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        th=float(self.theta); res={}
        for sym,g in df.groupby("symbol",sort=False):
            p=g.sort_index()["close"].to_numpy(float)
            n=p.size
            if n<2: res[sym]=np.nan; continue
            count=0; dir=0
            extreme=p[0]; ref=p[0]
            for px in p[1:]:
                if dir==0:
                    if px>=ref*(1+th): dir=+1; extreme=px; count+=1
                    elif px<=ref*(1-th): dir=-1; extreme=px; count+=1
                elif dir==+1:
                    if px>extreme: extreme=px
                    elif px<=extreme*(1-th): dir=-1; extreme=px; count+=1
                else:
                    if px<extreme: extreme=px
                    elif px>=extreme*(1+th): dir=+1; extreme=px; count+=1
            res[sym]=float(count)
        out["value"]=out["symbol"].map(res); return out

feature = DCCountBps10Feature()
