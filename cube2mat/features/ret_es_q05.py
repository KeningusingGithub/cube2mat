from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class RetESQ05Feature(BaseFeature):
    """
    5% Expected Shortfall (ES) of intraday log returns within 09:30–15:59.
      r = diff(log(close)); q = quantile(r,0.05); ES = mean(r[r<=q])
    NaN if <3 returns or tail empty.
    """
    name = "ret_es_q05"
    description = "5% ES (expected shortfall) of intraday log returns (RTH)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy()

        if df.empty or sample.empty:
            out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df["symbol"].isin(sample["symbol"].unique())].copy()
        df["close"]=pd.to_numeric(df["close"],errors="coerce")
        df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            r=np.log(g.sort_index()["close"]).diff().replace([np.inf,-np.inf],np.nan).dropna()
            if len(r)<3: res[sym]=np.nan; continue
            q=float(r.quantile(0.05))
            tail=r[r<=q]
            res[sym]=float(tail.mean()) if len(tail)>0 else np.nan
        out["value"]=out["symbol"].map(res); return out


feature = RetESQ05Feature()
