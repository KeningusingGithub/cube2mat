from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class PostVolRatioToRTHFeature(BaseFeature):
    """
    Ratio of post-market total volume to RTH total volume:
      Post: 16:00–23:59; RTH: 09:30–15:59.
    NaN if RTH volume<=0.
    """
    name = "post_vol_ratio_to_rth"
    description = "Post-market volume / RTH volume ratio for the day."
    required_full_columns = ("symbol","time","volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","volume"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()

        if df.empty or sample.empty:
            out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz)
        df=df[df["symbol"].isin(sample["symbol"].unique())].copy()
        df["volume"]=pd.to_numeric(df["volume"],errors="coerce"); df=df.dropna(subset=["volume"])
        if df.empty: out["value"]=pd.NA; return out

        res={}
        post=df.between_time("16:00","23:59")
        rth =df.between_time("09:30","15:59")
        for sym in sample["symbol"].unique():
            v_post=float(post[post["symbol"]==sym]["volume"].sum())
            v_rth =float(rth [rth ["symbol"]==sym]["volume"].sum())
            res[sym]= (v_post/v_rth) if (np.isfinite(v_rth) and v_rth>0) else np.nan
        out["value"]=out["symbol"].map(res); return out


feature = PostVolRatioToRTHFeature()
