from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class TransitionAsymmetryFeature(BaseFeature):
    """
    Asymmetry of sign transitions using simple returns in RTH:
      value = P(up->up) - P(down->down)
    where P(up->up) computed among pairs with previous sign=+1 (exclude zeros), etc.
    """
    name = "transition_asymmetry"
    description = "P(up→up) - P(down→down) from simple-return signs (exclude zeros)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
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
            a = np.sign(s[:-1]); b = np.sign(s[1:])
            # up->?
            m_up = (a==1)
            p_uu = np.nan
            if m_up.sum()>0:
                p_uu = float((b[m_up]==1).mean())
            # down->?
            m_dn = (a==-1)
            p_dd = np.nan
            if m_dn.sum()>0:
                p_dd = float((b[m_dn]==-1).mean())
            val = np.nan
            if np.isfinite(p_uu) and np.isfinite(p_dd):
                val = p_uu - p_dd
            res[sym] = val
        out["value"]=out["symbol"].map(res); return out

feature = TransitionAsymmetryFeature()
