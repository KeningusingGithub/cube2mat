# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df): return df.between_time("09:30","15:59")
def _logret(s): return np.log(s.astype(float)).diff()

def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.Series(a).astype(float); b = pd.Series(b).astype(float)
    mask = a.notna() & b.notna()
    if mask.sum() < 2: return float("nan")
    return float(np.corrcoef(a[mask].values, b[mask].values)[0,1])

class CorrSignRetVolume(BaseFeature):
    name = "corr_signret_volume"
    description = "Pearson correlation between sign(log-return) and volume within RTH (lag 0)."
    required_full_columns = ("symbol","time","close","volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx,date,columns=list(self.required_full_columns))
        pv = self.load_pv(ctx,date,columns=["symbol"])
        if df is None or pv is None: return None
        df = self.ensure_et_index(df, time_col="time", tz=ctx.tz)

        out={}
        for sym,g in df.groupby("symbol", observed=True):
            try:
                gg = _rth(g)[["close","volume"]].dropna()
                r = _logret(gg["close"]).dropna()
                vol = gg["volume"].astype(float).reindex(r.index)
                out[sym] = _safe_corr(np.sign(r), vol)
            except Exception:
                out[sym]=float("nan")
        res = pv[["symbol"]].copy(); res["value"] = res["symbol"].map(out); return res

feature = CorrSignRetVolume()
