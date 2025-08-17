from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class CorrAbsRetNFeature(BaseFeature):
    """
    |log 收益| 与成交笔数 n 的皮尔逊相关：corr(|r|, n)。
    """
    name = "corr_absret_n"
    description = "Pearson correlation between |log returns| and trade count n."
    required_full_columns = ("symbol","time","close","n")
    required_pv_columns   = ("symbol",)

    @staticmethod
    def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
        n = len(x)
        if n < 2: return np.nan
        xd = x - x.mean(); yd = y - y.mean()
        sxx = (xd*xd).sum(); syy = (yd*yd).sum()
        if sxx <= 0 or syy <= 0: return np.nan
        return float((xd*yd).sum()/np.sqrt(sxx*syy))

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full=self.load_full(ctx,date,list(self.required_full_columns))
        sample=self.load_pv(ctx,date,list(self.required_pv_columns))
        if full is None or sample is None: return None

        out=sample[["symbol"]].copy()
        if full.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(full,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty: out["value"]=pd.NA; return out

        for c in ("close","n"):
            df[c]=pd.to_numeric(df[c], errors="coerce")
        df=df[(df["close"]>0)].dropna(subset=["close","n"]).sort_index()
        if df.empty: out["value"]=pd.NA; return out

        df["log_close"]=np.log(df["close"])
        df["r"]=df.groupby("symbol",sort=False)["log_close"].diff().replace([np.inf,-np.inf],np.nan)
        df=df.dropna(subset=["r"])
        if df.empty: out["value"]=pd.NA; return out

        df["absr"]=df["r"].abs()
        value = df.groupby("symbol").apply(lambda g: self._pearson_corr(g["absr"], g["n"]))
        out["value"]=out["symbol"].map(value)
        return out

feature = CorrAbsRetNFeature()
