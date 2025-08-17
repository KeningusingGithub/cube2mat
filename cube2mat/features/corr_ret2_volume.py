from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class CorrRet2VolumeFeature(BaseFeature):
    """
    r^2 与 volume 的皮尔逊相关：corr(r_t^2, volume_t)，衡量量与瞬时方差的联动。
    """
    name = "corr_ret2_volume"
    description = "Pearson correlation between squared log returns and volume."
    required_full_columns = ("symbol","time","close","volume")
    required_pv_columns   = ("symbol",)

    @staticmethod
    def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
        if len(x) < 2: return np.nan
        xd = x - x.mean(); yd = y - y.mean()
        sxx = (xd*xd).sum(); syy = (yd*yd).sum()
        if sxx <= 0 or syy <= 0: return np.nan
        return float((xd*yd).sum()/np.sqrt(sxx*syy))

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full=self.load_full(ctx,date,list(self.required_full_columns))
        sample=self.load_pv(ctx,date,list(self.required_pv_columns))
        if full is None or sample is None: return None

        out=sample[["symbol"]].copy()
        if full.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(full,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty: out["value"]=pd.NA; return out

        for c in ("close","volume"):
            df[c]=pd.to_numeric(df[c], errors="coerce")
        df=df[(df["close"]>0)].dropna(subset=["close","volume"]).sort_index()
        if df.empty: out["value"]=pd.NA; return out

        df["log_close"]=np.log(df["close"])
        df["r"]=df.groupby("symbol",sort=False)["log_close"].diff().replace([np.inf,-np.inf],np.nan)
        df=df.dropna(subset=["r"])
        if df.empty: out["value"]=pd.NA; return out

        df["r2"]=df["r"]*df["r"]
        value = df.groupby("symbol").apply(lambda g: self._pearson_corr(g["r2"], g["volume"]))
        out["value"]=out["symbol"].map(value)
        return out

feature = CorrRet2VolumeFeature()
