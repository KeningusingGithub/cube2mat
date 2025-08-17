from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class RetExcessKurtosisFeature(BaseFeature):
    """
    log 收益的样本修正超额峰度（fat tails）：
      g2 = [n(n+1)/((n-1)(n-2)(n-3))]*m4/s^2 - [3(n-1)^2/((n-2)(n-3))]
      其中 s^2 为样本方差(ddof=1)，m4 = (1/n) * sum((r-mean)^4)。
    n<4 或 s^2<=0 则 NaN。
    """
    name = "ret_excess_kurtosis"
    description = "Sample-adjusted excess kurtosis of intraday log returns."
    required_full_columns = ("symbol","time","close")
    required_pv_columns   = ("symbol",)

    @staticmethod
    def _excess_kurt(s: pd.Series) -> float:
        r = s.values.astype(float)
        n = len(r)
        if n < 4: return np.nan
        m = r.mean()
        d = r - m
        s2 = np.sum(d*d)/(n-1)
        if s2 <= 0: return np.nan
        m4 = np.sum(d**4)/n
        g2 = (n*(n+1))/((n-1)*(n-2)*(n-3)) * (m4/(s2*s2)) - (3*(n-1)*(n-1))/((n-2)*(n-3))
        return float(g2)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full=self.load_full(ctx,date,list(self.required_full_columns))
        sample=self.load_pv(ctx,date,list(self.required_pv_columns))
        if full is None or sample is None: return None

        out=sample[["symbol"]].copy()
        if full.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(full,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty: out["value"]=pd.NA; return out

        df["close"]=pd.to_numeric(df["close"], errors="coerce")
        df=df[(df["close"]>0)].dropna(subset=["close"]).sort_index()
        if df.empty: out["value"]=pd.NA; return out

        df["log_close"]=np.log(df["close"])
        df["r"]=df.groupby("symbol",sort=False)["log_close"].diff().replace([np.inf,-np.inf],np.nan)
        df=df.dropna(subset=["r"])
        if df.empty: out["value"]=pd.NA; return out

        value=df.groupby("symbol")["r"].apply(self._excess_kurt)
        out["value"]=out["symbol"].map(value)
        return out

feature = RetExcessKurtosisFeature()
