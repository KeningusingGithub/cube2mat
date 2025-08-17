# features/oc_return_last30m_share.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class OCReturnLast30mShareFeature(BaseFeature):
    """
    Share of net RTH log return contributed by last 30 minutes (15:30–15:59):
      share = sum r_tail / sum r_all. NaN if denominator==0 or insufficient.
    """
    name = "oc_return_last30m_share"
    description = "Net-return share from last 30 minutes of RTH."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()

        df=self.ensure_et_index(df,"time",ctx.tz)
        rth=df.between_time("09:30","15:59")
        tail=df.between_time("15:30","15:59")
        rth=rth[rth.symbol.isin(sample.symbol.unique())].copy()
        for c in ("close",): rth[c]=pd.to_numeric(rth[c],errors="coerce")
        rth=rth.dropna(subset=["close"])
        if rth.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in rth.groupby("symbol",sort=False):
            g=g.sort_index()
            r=np.log(g["close"]).diff().replace([np.inf,-np.inf],np.nan).dropna()
            if r.empty: res[sym]=np.nan; continue
            total=float(r.sum())
            if total==0: res[sym]=np.nan; continue
            r.index=g.index[1:]
            part=r[r.index.isin(tail.index)]
            res[sym]=float(part.sum()/total) if not part.empty else np.nan
        out["value"]=out["symbol"].map(res); return out

feature = OCReturnLast30mShareFeature()
