from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class CVVolumeFeature(BaseFeature):
    """
    Coefficient of variation of per-bar volume in RTH: std(volume)/mean(volume).
    NaN if <3 bars or mean<=0.
    """
    name = "cv_volume"
    description = "Std/mean of volume across RTH bars."
    required_full_columns = ("symbol","time","volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","volume"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()

        if df.empty or sample.empty:
            out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df["symbol"].isin(sample["symbol"].unique())].copy()
        df["volume"]=pd.to_numeric(df["volume"],errors="coerce")
        df=df.dropna(subset=["volume"])

        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            v=g.sort_index()["volume"].dropna()
            if len(v)<3: res[sym]=np.nan; continue
            mu=float(v.mean()); sd=float(v.std(ddof=1))
            res[sym]=sd/mu if (np.isfinite(mu) and mu>0) else np.nan
        out["value"]=out["symbol"].map(res); return out


feature = CVVolumeFeature()
