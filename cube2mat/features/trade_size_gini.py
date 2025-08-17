from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext
from volume_gini import _gini  # 复用


class TradeSizeGiniFeature(BaseFeature):
    """
    Gini index of per-bar trade size (volume/n) distribution in RTH.
    Use bars with n>0; NaN if <2 valid bars or sum(tsize)<=0.
    """
    name = "trade_size_gini"
    description = "Gini of per-bar average trade size (volume/n) across RTH bars."
    required_full_columns = ("symbol","time","volume","n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","volume","n"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()

        if df.empty or sample.empty:
            out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df["symbol"].isin(sample["symbol"].unique())].copy()
        df["volume"]=pd.to_numeric(df["volume"],errors="coerce")
        df["n"]=pd.to_numeric(df["n"],errors="coerce")
        df=df.dropna(subset=["volume","n"])
        df=df[df["n"]>0]

        if df.empty: out["value"]=pd.NA; return out
        res = df.assign(tsize=df["volume"]/df["n"]).groupby("symbol")["tsize"].apply(lambda s: _gini(s.values))
        out["value"]=out["symbol"].map(res); return out


feature = TradeSizeGiniFeature()
