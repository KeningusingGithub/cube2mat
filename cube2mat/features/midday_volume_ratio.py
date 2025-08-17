from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class MiddayVolumeRatioFeature(BaseFeature):
    """
    Mean(volume) in 12:00–12:59 divided by mean(volume) over entire RTH.
    NaN if denominators invalid or segments empty.
    """
    name = "midday_volume_ratio"
    description = "Midday (12-13h) mean volume / session mean volume."
    required_full_columns = ("symbol","time","volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","volume"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        all_=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        mid=self.ensure_et_index(df,"time",ctx.tz).between_time("12:00","12:59").copy()
        for part in (all_,mid): part["volume"]=pd.to_numeric(part["volume"],errors="coerce")
        all_=all_.dropna(subset=["volume"]); mid=mid.dropna(subset=["volume"])
        all_=all_[all_.symbol.isin(sample.symbol.unique())]
        mid=mid[mid.symbol.isin(sample.symbol.unique())]
        if all_.empty: out["value"]=pd.NA; return out
        res={}
        for sym in sample["symbol"].unique():
            v_all=all_[all_["symbol"]==sym]["volume"]
            if v_all.empty or v_all.mean()<=0: res[sym]=np.nan; continue
            v_mid=mid[mid["symbol"]==sym]["volume"]
            if v_mid.empty: res[sym]=np.nan; continue
            res[sym]=float(v_mid.mean()/v_all.mean())
        out["value"]=out["symbol"].map(res); return out

feature = MiddayVolumeRatioFeature()
