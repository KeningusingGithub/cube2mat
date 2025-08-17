from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class CorrAbsRetVolumeFeature(BaseFeature):
    """
    |log 收益| 与成交量的皮尔逊相关：corr(|r|, volume)。
    对齐为“收益时刻”的体量（即差分后的后一个 bar 的 volume）。
    样本<2 或方差为 0 则 NaN。
    """
    name = "corr_absret_volume"
    description = "Pearson correlation between |log returns| and volume within session."
    required_full_columns = ("symbol","time","close","volume")
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

        for c in ("close","volume"):
            df[c]=pd.to_numeric(df[c], errors="coerce")
        df=df[(df["close"]>0)].dropna(subset=["close","volume"]).sort_index()
        if df.empty: out["value"]=pd.NA; return out

        df["log_close"]=np.log(df["close"])
        df["r"]=df.groupby("symbol",sort=False)["log_close"].diff().replace([np.inf,-np.inf],np.nan)
        df=df.dropna(subset=["r"])
        if df.empty: out["value"]=pd.NA; return out

        df["absr"]=df["r"].abs()
        # 对齐 volume（收益时刻为“后一个 bar”时间戳）
        df = df.dropna(subset=["absr","volume"])
        value = df.groupby("symbol").apply(lambda g: self._pearson_corr(g["absr"], g["volume"]))
        out["value"]=out["symbol"].map(value)
        return out

feature = CorrAbsRetVolumeFeature()
