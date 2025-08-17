from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class MarubozuDensityFeature(BaseFeature):
    """
    Share of 'marubozu-like' bars:
      upper_shadow_ratio <= alpha AND lower_shadow_ratio <= alpha AND body_ratio >= beta
    Defaults: alpha=0.10, beta=0.80; exclude zero-range bars.
    """
    name = "marubozu_density"
    description = "Density of long-body, tiny-shadow bars (alpha=0.10, beta=0.80) in RTH."
    required_full_columns = ("symbol","time","open","high","low","close")
    required_pv_columns = ("symbol",)
    alpha = 0.10
    beta  = 0.80

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
        body=(df["close"]-df["open"]).abs()
        up=(df["high"]-np.maximum(df["open"],df["close"])).clip(lower=0)
        lowsh=(np.minimum(df["open"],df["close"]) - df["low"]).clip(lower=0)
        df2=df.assign(rng=rng, body=body, up=up, lowsh=lowsh)
        df2=df2[df2["rng"]>0]

        a=float(self.alpha); b=float(self.beta)
        cond = (df2["up"]/df2["rng"] <= a) & (df2["lowsh"]/df2["rng"] <= a) & (df2["body"]/df2["rng"] >= b)
        res = df2.groupby("symbol").apply(lambda g: float(cond.loc[g.index].mean()) if len(g)>0 else np.nan)
        out["value"]=out["symbol"].map(res); return out

feature = MarubozuDensityFeature()
