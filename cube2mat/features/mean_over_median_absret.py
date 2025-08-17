from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class MeanOverMedianAbsRetFeature(BaseFeature):
    """
    Ratio mean(|logret|)/median(|logret|) within 09:30–15:59.
    NaN if <3 returns or median==0.
    """
    name = "mean_over_median_absret"
    description = "mean(|logret|)/median(|logret|) as tail-sensitivity proxy (RTH)."
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
            a=np.log(g.sort_index()["close"]).diff().replace([np.inf,-np.inf],np.nan).abs().dropna()
            if len(a)<3: res[sym]=np.nan; continue
            med=float(a.median())
            mu=float(a.mean())
            res[sym]=mu/med if (np.isfinite(med) and med>0) else np.nan
        out["value"]=out["symbol"].map(res); return out


feature = MeanOverMedianAbsRetFeature()
