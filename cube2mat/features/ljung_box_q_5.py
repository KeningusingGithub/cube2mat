from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext
from cube2mat.features.ac_ret_lag2 import _acf_at_lag

class LjungBoxQ5Feature(BaseFeature):
    """
    Ljung-Box Q statistic for log returns using lags 1..5:
      Q = n(n+2) * sum_{k=1..5} rho_k^2 / (n-k)
    Returns Q; NaN if n<=5 or any denominator invalid.
    """
    name = "ljung_box_q_5"
    description = "Ljung-Box Q statistic at lags 1..5 for intraday log returns."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)
    m = 5

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx,date,list(self.required_full_columns))
        sample = self.load_pv(ctx,date,list(self.required_pv_columns))
        if df is None or sample is None: return None
        out = sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df = self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df = df[df.symbol.isin(set(sample.symbol.unique()))].copy()
        df["close"]=pd.to_numeric(df["close"],errors="coerce")
        df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        m = int(self.m); res={}
        for sym,g in df.groupby("symbol",sort=False):
            r = np.log(g.sort_index()["close"]).diff().replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
            n = r.size
            if n <= m: res[sym]=np.nan; continue
            q = 0.0; valid = True
            for k in range(1, m+1):
                rho = _acf_at_lag(r, k)
                if not np.isfinite(rho) or (n - k) <= 0:
                    valid = False; break
                q += (rho*rho) / (n - k)
            if not valid:
                res[sym] = np.nan
            else:
                res[sym] = float(n * (n + 2) * q)
        out["value"]=out["symbol"].map(res); return out

feature = LjungBoxQ5Feature()
