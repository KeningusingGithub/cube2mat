from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class VarConcTopDecileFeature(BaseFeature):
    """
    方差贡献集中度（Top 10%）：RV_top10 / RV_total，其中
      RV_total = sum(r^2)，r=diff(log(close))；
      RV_top10 = 对 r^2 的 90% 分位数阈值以上部分的总和。
    分母<=0 或有效 r<2 则 NaN。
    """
    name = "var_conc_top_decile"
    description = "Share of realized variance contributed by the top 10% largest r^2."
    required_full_columns = ("symbol","time","close")
    required_pv_columns   = ("symbol",)

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

        def per_symbol(g: pd.DataFrame) -> float:
            r2 = (g["r"]*g["r"]).values
            if r2.size < 2: return np.nan
            total = float(r2.sum())
            if total <= 0: return np.nan
            thr = np.quantile(r2, 0.9)
            top = float(r2[r2 >= thr].sum())
            return top/total

        value = df.groupby("symbol").apply(per_symbol)
        out["value"]=out["symbol"].map(value)
        return out

feature = VarConcTopDecileFeature()
