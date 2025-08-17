from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class RogersSatchellSessionVolFeature(BaseFeature):
    """
    会话级 Rogers–Satchell 波动率（基于 O/H/L/C 聚合）：
      var_RS = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
      vol    = sqrt(max(var_RS, 0))
    要求 O,H,L,C > 0，否则 NaN。
    """
    name = "rogers_satchell_session_vol"
    description = "Session-level Rogers–Satchell volatility using aggregated O/H/L/C."
    required_full_columns = ("symbol", "time", "open", "high", "low", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx,date,list(self.required_full_columns))
        sample = self.load_pv(ctx,date,list(self.required_pv_columns))
        if full is None or sample is None: return None
        out = sample[["symbol"]].copy()
        if full.empty or sample.empty: out["value"]=pd.NA; return out

        df  = self.ensure_et_index(full,"time",ctx.tz).between_time("09:30","15:59")
        df  = df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty: out["value"]=pd.NA; return out

        for c in ("open","high","low","close"):
            df[c]=pd.to_numeric(df[c], errors="coerce")
        df=df.dropna(subset=["open","high","low","close"])
        if df.empty: out["value"]=pd.NA; return out
        df=df.sort_index()

        def per_symbol(g: pd.DataFrame) -> float:
            O = g["open"].dropna().iloc[0]
            C = g["close"].dropna().iloc[-1]
            H = g["high"].max()
            L = g["low"].min()
            if not all(np.isfinite(x) and x>0 for x in (O,H,L,C)): return np.nan
            v = np.log(H/C)*np.log(H/O) + np.log(L/C)*np.log(L/O)
            if not np.isfinite(v): return np.nan
            v = max(v, 0.0)
            return float(np.sqrt(v))

        value = df.groupby("symbol").apply(per_symbol)
        out["value"]=out["symbol"].map(value)
        return out

feature = RogersSatchellSessionVolFeature()
