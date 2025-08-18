# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df): return df.between_time("09:30", "15:59")
def _logret(s): return np.log(s.astype(float)).diff()

class RangeToRVSqrtRatio(BaseFeature):
    name = "range_to_rv_sqrt_ratio"
    description = "Log session range log(high_max/low_min) divided by sqrt(Σ r^2) using RTH log-returns."
    required_full_columns = ("symbol", "time", "high", "low", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=list(self.required_full_columns))
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None: return None
        df = self.ensure_et_index(df, time_col="time", tz=ctx.tz)

        out={}
        for sym,g in df.groupby("symbol", observed=True):
            try:
                d = _rth(g)[["high","low","close"]].dropna()
                if d.empty: out[sym]=float("nan"); continue
                rng = float(np.log(d["high"].max()/d["low"].min()))
                r = _logret(d["close"]).dropna().values
                denom = float(np.sqrt(np.sum(r**2)))
                out[sym] = float(rng/denom) if np.isfinite(denom) and denom>0 else float("nan")
            except Exception:
                out[sym]=float("nan")
        res = pv[["symbol"]].copy(); res["value"]=res["symbol"].map(out); return res

feature = RangeToRVSqrtRatio()
