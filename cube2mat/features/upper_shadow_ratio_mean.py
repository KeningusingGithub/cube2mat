from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class UpperShadowRatioMeanFeature(BaseFeature):
    """
    Mean upper shadow ratio: (high - max(open,close)) / (high - low), exclude zero-range bars.
    """
    name = "upper_shadow_ratio_mean"
    description = "Mean upper shadow / range ratio in RTH; exclude zero-range bars."
    required_full_columns = ("symbol","time","open","high","low","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,list(self.required_full_columns))
        sample=self.load_pv(ctx,date,list(self.required_pv_columns))
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(set(sample.symbol.unique()))]
        if df.empty: out["value"]=pd.NA; return out
        for c in ("open","high","low","close"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["open","high","low","close"])
        if df.empty: out["value"]=pd.NA; return out

        rng=(df["high"]-df["low"])
        up=(df["high"]-np.maximum(df["open"],df["close"])).clip(lower=0)
        df2=df.assign(rng=rng, up=up)
        df2=df2[df2["rng"]>0]
        res=df2.groupby("symbol").apply(lambda g: float((g["up"]/g["rng"]).mean()) if len(g)>0 else np.nan)
        out["value"]=out["symbol"].map(res); return out

feature = UpperShadowRatioMeanFeature()
