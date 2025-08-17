from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class TransitionPSameSignFeature(BaseFeature):
    """
    P(sign_t == sign_{t-1}) using simple returns in RTH, excluding zeros.
    """
    name = "transition_p_same_sign"
    description = "Probability that consecutive simple returns have the same sign (exclude zeros)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,list(self.required_full_columns))
        sample=self.load_pv(ctx,date,list(self.required_pv_columns))
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(set(sample.symbol.unique()))].copy()
        df["close"]=pd.to_numeric(df["close"],errors="coerce")
        df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            s = g.sort_index()["close"].pct_change().to_numpy()
            sgn = np.sign(s)
            # valid pairs where both signs are nonzero
            a = sgn[:-1]; b = sgn[1:]
            mask = (a!=0) & (b!=0)
            if mask.sum() < 1: res[sym]=np.nan; continue
            match = (a[mask] == b[mask]).mean()
            res[sym] = float(match)
        out["value"]=out["symbol"].map(res); return out

feature = TransitionPSameSignFeature()
