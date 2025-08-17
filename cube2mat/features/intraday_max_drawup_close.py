from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class IntradayMaxDrawupCloseFeature(BaseFeature):
    """
    Maximum intraday drawup on close in RTH:
      max_t (close_t / running_min_up_to_t - 1). NaN if <2 closes.
    """
    name = "intraday_max_drawup_close"
    description = "Max drawup using close (RTH)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy(); 
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        df=df[df.symbol.isin(sample.symbol.unique())]
        df["close"]=pd.to_numeric(df["close"],errors="coerce")
        df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out
        res={}
        for sym,g in df.groupby("symbol",sort=False):
            c=g.sort_index()["close"].to_numpy(float)
            n=c.size
            if n<2: res[sym]=np.nan; continue
            run_min=c[0]; maxdu=0.0
            for v in c[1:]:
                run_min=min(run_min, v)
                if run_min>0:
                    du=v/run_min - 1.0
                    if np.isfinite(du): maxdu=max(maxdu, du)
            res[sym]=float(maxdu)
        out["value"]=out["symbol"].map(res); return out

feature = IntradayMaxDrawupCloseFeature()
