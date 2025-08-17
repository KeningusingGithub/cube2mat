from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class VolumeEntropyFeature(BaseFeature):
    """
    Normalized Shannon entropy H/Hmax of volume distribution across RTH bars, in [0,1].
    p_i = v_i/sum(v); H = -sum p_i log p_i; Hmax=log(m). NaN if sum<=0 or m<2.
    """
    name = "volume_entropy"
    description = "Normalized entropy of per-bar volume distribution (RTH)."
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
            v = g.sort_index()["volume"].astype(float).values
            s = float(np.nansum(v))
            if not np.isfinite(s) or s<=0: res[sym]=np.nan; continue
            p = v / s
            p = p[p>0]
            m = p.size
            if m < 2: res[sym]=np.nan; continue
            H = float(-(p*np.log(p)).sum())
            val = H / np.log(m)
            res[sym]=float(np.clip(val,0.0,1.0))
        out["value"]=out["symbol"].map(res); return out


feature = VolumeEntropyFeature()
