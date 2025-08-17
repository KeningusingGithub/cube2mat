# features/path_length_over_range.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class PathLengthOverRangeFeature(BaseFeature):
    """
    Path choppiness ratio: sum(|Δclose|) / (max(high)-min(low)) within RTH.
    NaN if <2 closes or range<=0.
    """
    name = "path_length_over_range"
    description = "Sum|Δclose| normalized by session range (H-L)."
    required_full_columns = ("symbol","time","high","low","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","high","low","close"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("high","low","close"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["high","low","close"])
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            c=g["close"].to_numpy(float)
            if c.size<2: res[sym]=np.nan; continue
            path=float(np.abs(np.diff(c)).sum())
            H=float(g["high"].max()); L=float(g["low"].min())
            rng=H-L
            res[sym]= path/rng if rng>0 else np.nan
        out["value"]=out["symbol"].map(res); return out

feature = PathLengthOverRangeFeature()
