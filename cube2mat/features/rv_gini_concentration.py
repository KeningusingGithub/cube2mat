from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class RVGiniConcentrationFeature(BaseFeature):
    """
    RV 的 Gini 集中度（r^2 的不均衡程度）：
      对 y_i = r_i^2（i=1..n），按升序排序：
      G = [sum_{i=1}^n (2i - n - 1) * y_i] / [n * sum y_i]，取值 ∈ [0,1)；越大越集中。
    若 sum y_i <= 0 或 n<2 则 NaN。
    """
    name = "rv_gini_concentration"
    description = "Gini concentration of realized variance contributions r^2."
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

        def per_symbol(s: pd.Series) -> float:
            y = (s.values**2).astype(float)
            n = y.size
            if n < 2: return np.nan
            tot = y.sum()
            if not np.isfinite(tot) or tot <= 0: return np.nan
            y.sort()
            i = np.arange(1, n+1, dtype=float)
            G = np.sum((2*i - n - 1) * y) / (n * tot)
            return float(G)

        value = df.groupby("symbol")["r"].apply(per_symbol)
        out["value"]=out["symbol"].map(value)
        return out

feature = RVGiniConcentrationFeature()
