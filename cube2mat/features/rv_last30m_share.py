from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class RVLast30mShareFeature(BaseFeature):
    """
    末 30 分钟已实现方差占比：
      share = sum_{t in last30m}(r_t^2) / sum_{all}(r_t^2)
    r_t = diff(log(close))；close>0。若分母<=0 则 NaN。
    """
    name = "rv_last30m_share"
    description = "Share of realized variance contributed by the last 30 minutes of the session."
    required_full_columns = ("symbol","time","close")
    required_pv_columns   = ("symbol",)
    TOTAL_MIN = (pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")).total_seconds()/60.0

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full=self.load_full(ctx,date,list(self.required_full_columns))
        sample=self.load_pv(ctx,date,list(self.required_pv_columns))
        if full is None or sample is None: return None

        out=sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"]=pd.NA; return out

        df=self.ensure_et_index(full,"time",ctx.tz).between_time("09:30","15:59").sort_index()
        df=df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty:
            out["value"]=pd.NA; return out

        df["close"]=pd.to_numeric(df["close"], errors="coerce")
        df=df[(df["close"]>0)].dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        df["log_close"]=np.log(df["close"])
        df["r"]=df.groupby("symbol",sort=False)["log_close"].diff().replace([np.inf,-np.inf],np.nan)
        df=df.dropna(subset=["r"])
        if df.empty:
            out["value"]=pd.NA; return out

        tmin = (df.index - df.index.normalize() - pd.Timedelta("09:30:00")).total_seconds()/60.0
        df["t_min"] = tmin

        def per_symbol(g: pd.DataFrame) -> float:
            r2 = (g["r"]*g["r"])
            denom = float(r2.sum())
            if denom <= 0: return np.nan
            last = r2[g["t_min"] >= (self.TOTAL_MIN - 30.0)].sum()
            return float(last/denom)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"]=out["symbol"].map(value)
        return out

feature = RVLast30mShareFeature()
