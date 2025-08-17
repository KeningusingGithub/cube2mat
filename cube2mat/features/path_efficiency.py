from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class PathEfficiencyFeature(BaseFeature):
    """
    Path efficiency in [0,1]:
      |last_close - first_open| / sum(|Δclose|)
    Uses first available open within RTH as anchor; NaN if <2 closes or denom<=0.
    """
    name = "path_efficiency"
    description = "Path straightness: |last_close - first_open| / sum|Δclose| in RTH."
    required_full_columns = ("symbol","time","open","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,list(self.required_full_columns))
        sample=self.load_pv(ctx,date,list(self.required_pv_columns))
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(set(sample.symbol.unique()))].copy()
        for c in ("open","close"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            closes=g["close"].dropna()
            if len(closes)<2: res[sym]=np.nan; continue
            first_open = g["open"].dropna()
            if first_open.empty: res[sym]=np.nan; continue
            anchor = float(first_open.iloc[0])
            lastc = float(closes.iloc[-1])
            denom = float(np.abs(closes.diff().dropna()).sum())
            if not np.isfinite(denom) or denom<=0:
                res[sym]=np.nan; continue
            val = abs(lastc - anchor) / denom
            res[sym] = float(np.clip(val, 0.0, 1.0))
        out["value"]=out["symbol"].map(res); return out

feature = PathEfficiencyFeature()
