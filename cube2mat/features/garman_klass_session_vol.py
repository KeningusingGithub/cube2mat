from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class GarmanKlassSessionVolFeature(BaseFeature):
    """
    会话级 Garman–Klass 波动率（基于本日会话的 O/H/L/C 聚合）：
      O = 会话首笔 open、C = 会话末笔 close、H = 会话内 max(high)、L = 会话内 min(low)
      var_GK = 0.5*[ln(H/L)]^2 - (2ln2 - 1)*[ln(C/O)]^2
      vol    = sqrt(max(var_GK, 0))
    要求 O,H,L,C > 0，否则 NaN。
    """
    name = "garman_klass_session_vol"
    description = "Session-level Garman–Klass volatility using aggregated O/H/L/C."
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
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open","high","low","close"])
        if df.empty: out["value"]=pd.NA; return out
        df = df.sort_index()

        def per_symbol(g: pd.DataFrame) -> float:
            O = g["open"].dropna().iloc[0]
            C = g["close"].dropna().iloc[-1]
            H = g["high"].max()
            L = g["low"].min()
            if not all(np.isfinite(x) and x>0 for x in (O,H,L,C)): return np.nan
            lnHL = np.log(H/L)
            lnCO = np.log(C/O)
            var_gk = 0.5*(lnHL*lnHL) - (2.0*np.log(2.0)-1.0)*(lnCO*lnCO)
            if not np.isfinite(var_gk): return np.nan
            var_gk = max(var_gk, 0.0)
            return float(np.sqrt(var_gk))

        value = df.groupby("symbol").apply(per_symbol)
        out["value"]=out["symbol"].map(value)
        return out

feature = GarmanKlassSessionVolFeature()
