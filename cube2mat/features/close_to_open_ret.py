from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class CloseToOpenRetFeature(BaseFeature):
    """
    Session close-to-open simple return (reverse anchor within same RTH):
      value = first_open / last_close - 1; NaN if missing or last_close<=0.
    """
    name = "close_to_open_ret"
    description = "Simple return from last close back to first open within same RTH."
    required_full_columns = ("symbol","time","open","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","open","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy()

        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("open","close"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["open","close"])
        df=df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            o=g["open"].dropna()
            c=g["close"].dropna()
            if o.empty or c.empty: res[sym]=np.nan; continue
            o0=float(o.iloc[0]); cL=float(c.iloc[-1])
            if cL<=0: res[sym]=np.nan; continue
            res[sym]= o0/cL - 1.0
        out["value"]=out["symbol"].map(res); return out


feature = CloseToOpenRetFeature()
