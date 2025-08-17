from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class PremktVolRatioToRTHFeature(BaseFeature):
    """
    Ratio of pre-market total volume to RTH total volume:
      Pre: 00:00–09:29; RTH: 09:30–15:59.
    NaN if RTH volume<=0.
    """
    name = "premkt_vol_ratio_to_rth"
    description = "Pre-market volume / RTH volume ratio for the day."
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
        pre=df.between_time("00:00","09:29")
        rth=df.between_time("09:30","15:59")
        for sym in sample["symbol"].unique():
            v_pre=float(pre[pre["symbol"]==sym]["volume"].sum())
            v_rth=float(rth[rth["symbol"]==sym]["volume"].sum())
            res[sym]= (v_pre/v_rth) if (np.isfinite(v_rth) and v_rth>0) else np.nan
        out["value"]=out["symbol"].map(res); return out


feature = PremktVolRatioToRTHFeature()
