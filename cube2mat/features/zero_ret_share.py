from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class ZeroRetShareFeature(BaseFeature):
    """
    Share of bars with simple return close.pct_change()==0 (within epsilon).
    epsilon default=1e-10; NaN if <1 return.
    """
    name = "zero_ret_share"
    description = "Fraction of bars with near-zero simple returns (|ret|<=1e-10) in RTH."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)
    eps = 1e-10

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
        eps=float(self.eps)
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            r=g.sort_index()["close"].pct_change().replace([np.inf,-np.inf],np.nan).dropna()
            if len(r)<1: res[sym]=np.nan; continue
            res[sym]=float((r.abs()<=eps).mean())
        out["value"]=out["symbol"].map(res); return out


feature = ZeroRetShareFeature()
