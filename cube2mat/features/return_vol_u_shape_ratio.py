# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df): return df.between_time("09:30", "15:59")
def _logret(s: pd.Series) -> pd.Series: return np.log(s.astype(float)).diff()

class ReturnVolUShapeRatio(BaseFeature):
    name = "return_vol_u_shape_ratio"
    description = "((std r in 09:30–10:29 + std r in 15:00–15:59)/2) / std r in 11:00–14:00 (log-returns)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=list(self.required_full_columns))
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None: return None
        df = self.ensure_et_index(df, time_col="time", tz=ctx.tz)

        out={}
        for sym,g in df.groupby("symbol", observed=True):
            try:
                z = _rth(g)[["close"]].dropna()
                r = _logret(z["close"]).dropna()
                if r.empty: out[sym]=float("nan"); continue
                r_idx = r.index
                early = r.loc[_rth(g).between_time("09:30","10:29").index.intersection(r_idx)]
                mid   = r.loc[_rth(g).between_time("11:00","14:00").index.intersection(r_idx)]
                late  = r.loc[_rth(g).between_time("15:00","15:59").index.intersection(r_idx)]
                def _std(x): return float(np.std(x.values, ddof=1)) if x.size>=2 else np.nan
                e, m, l = _std(early), _std(mid), _std(late)
                out[sym] = float(((e+l)/2)/m) if np.isfinite(m) and m>0 and np.isfinite(e) and np.isfinite(l) else float("nan")
            except Exception:
                out[sym]=float("nan")

        res = pv[["symbol"]].copy(); res["value"]=res["symbol"].map(out); return res

feature = ReturnVolUShapeRatio()
