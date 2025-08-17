from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class CondReversalAfterRun2Feature(BaseFeature):
    """
    Probability of reversal after a 2-length run:
      events at t where sign(ret_{t-1})==sign(ret_t)!=0; outcome = 1[sign(ret_{t+1}) != sign(ret_t) and sign!=0].
    value = mean(outcome | events). NaN if <3 events.
    """
    name = "cond_reversal_after_run2"
    description = "P(reversal | two consecutive same-sign simple returns)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy(); 
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(sample.symbol.unique())].copy()
        df["close"]=pd.to_numeric(df["close"],errors="coerce"); df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            r=g.sort_index()["close"].pct_change().replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
            s=np.sign(r)
            if s.size<3: res[sym]=np.nan; continue
            a=s[:-2]; b=s[1:-1]; c=s[2:]
            events=(a==b) & (b!=0)        # two-length run ends at index of b
            valid_next=(c!=0)
            mask=events & valid_next
            if mask.sum()<3: res[sym]=np.nan; continue
            outcome=(c[mask]!=b[mask]).astype(float)
            res[sym]=float(outcome.mean())
        out["value"]=out["symbol"].map(res); return out

feature = CondReversalAfterRun2Feature()
