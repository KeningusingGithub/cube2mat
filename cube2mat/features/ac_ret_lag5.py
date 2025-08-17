from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext
from cube2mat.features.ac_ret_lag2 import _acf_at_lag

class ACRetLag5Feature(BaseFeature):
    """
    Autocorrelation of intraday log returns at lag=5 within 09:30–15:59.
    """
    name = "ac_ret_lag5"
    description = "Lag-5 autocorrelation of intraday log returns in RTH."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx,date,list(self.required_full_columns))
        sample = self.load_pv(ctx,date,list(self.required_pv_columns))
        if df is None or sample is None: return None
        out = sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df = self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df = df[df.symbol.isin(set(sample.symbol.unique()))].copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out
        res = {}
        for sym,g in df.groupby("symbol",sort=False):
            r = np.log(g.sort_index()["close"]).diff().replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
            res[sym] = float(_acf_at_lag(r, 5))
        out["value"] = out["symbol"].map(res)
        return out

feature = ACRetLag5Feature()
