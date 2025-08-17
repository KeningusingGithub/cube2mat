from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class OpenToCloseRetFeature(BaseFeature):
    """
    Session open-to-close simple return in RTH:
      value = last_close / first_open - 1; NaN if missing or nonpositive anchors.
    """
    name = "open_to_close_ret"
    description = "Simple return from first open to last close within RTH."
    required_full_columns = ("symbol","time","open","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","open","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy()

        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("open","close"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["open","close"])
        df=df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            o=g["open"].dropna()
            c=g["close"].dropna()
            if o.empty or c.empty: res[sym]=np.nan; continue
            o0=float(o.iloc[0]); cL=float(c.iloc[-1])
            if o0<=0: res[sym]=np.nan; continue
            res[sym]= cL/o0 - 1.0
        out["value"]=out["symbol"].map(res); return out


feature = OpenToCloseRetFeature()
