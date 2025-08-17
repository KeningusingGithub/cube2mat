from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class GKVarSumOHLCFeature(BaseFeature):
    """
    逐 bar 的 Garman–Klass 方差并在会话内求和（近似集成方差）：
      var_i = 0.5*[ln(H_i/L_i)]^2 - (2ln2 - 1)*[ln(C_i/O_i)]^2，按 bar 计算后 clip 至 >=0 再求和。
    要求各 bar 的 O/H/L/C > 0；若无有效 bar 或总和<=0 则 NaN。
    """
    name = "gk_var_sum_ohlc"
    description = "Sum of per-bar GK variance across the session (clipped at 0)."
    required_full_columns = ("symbol","time","open","high","low","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full=self.load_full(ctx,date,list(self.required_full_columns))
        sample=self.load_pv(ctx,date,list(self.required_pv_columns))
        if full is None or sample is None: return None
        out=sample[["symbol"]].copy()
        if full.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(full,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty: out["value"]=pd.NA; return out

        for c in ("open","high","low","close"):
            df[c]=pd.to_numeric(df[c], errors="coerce")
        df=df.dropna(subset=["open","high","low","close"])
        if df.empty: out["value"]=pd.NA; return out

        df = df[(df["open"]>0)&(df["high"]>0)&(df["low"]>0)&(df["close"]>0)].copy()
        if df.empty: out["value"]=pd.NA; return out

        lnHL = np.log(df["high"]/df["low"])
        lnCO = np.log(df["close"]/df["open"])
        df["gk_var"] = 0.5*(lnHL*lnHL) - (2.0*np.log(2.0)-1.0)*(lnCO*lnCO)
        df["gk_var"] = df["gk_var"].clip(lower=0.0)

        value = df.groupby("symbol")["gk_var"].sum()
        value = value.where(value>0)
        out["value"]=out["symbol"].map(value)
        return out

feature = GKVarSumOHLCFeature()
