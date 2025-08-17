from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class RetVaRQ05Feature(BaseFeature):
    """
    5% VaR of intraday log returns within 09:30–15:59.
    r = diff(log(close)); VaR_5 = quantile(r, 0.05). NaN if <3 returns.
    """
    name = "ret_var_q05"
    description = "5% quantile VaR of intraday log returns (RTH)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy()

        if df.empty or sample.empty:
            out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df["symbol"].isin(sample["symbol"].unique())].copy()
        df["close"]=pd.to_numeric(df["close"],errors="coerce")
        df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            r=np.log(g.sort_index()["close"]).diff().replace([np.inf,-np.inf],np.nan).dropna()
            res[sym]=float(r.quantile(0.05)) if len(r)>=3 else np.nan
        out["value"]=out["symbol"].map(res); return out


feature = RetVaRQ05Feature()
