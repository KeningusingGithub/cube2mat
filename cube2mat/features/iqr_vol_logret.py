from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class IQRVolLogRetFeature(BaseFeature):
    """
    基于 log 收益的稳健波动率（IQR 标定）：
      r = diff(log(close))；sigma_IQR ≈ 0.7413 * IQR(r) = 0.7413 * (Q75 - Q25)。
    有效 r 数 < 3 则 NaN。
    """
    name = "iqr_vol_logret"
    description = "Robust volatility via 0.7413*IQR of log returns."
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

        def per_symbol(s: pd.Series)->float:
            if len(s)<3: return np.nan
            q75 = np.quantile(s.values, 0.75)
            q25 = np.quantile(s.values, 0.25)
            return float(0.7413*(q75-q25))

        value=df.groupby("symbol")["r"].apply(per_symbol)
        out["value"]=out["symbol"].map(value)
        return out

feature = IQRVolLogRetFeature()
