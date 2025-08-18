# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df): return df.between_time("09:30","15:59")
def _logret(s): return np.log(s.astype(float)).diff()

class RetUpDownAbsMagnitudeRatio(BaseFeature):
    name = "ret_up_down_abs_magnitude_ratio"
    description = "Sum of positive |r| divided by sum of negative |r| for RTH log-returns."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx,date,columns=list(self.required_full_columns))
        pv = self.load_pv(ctx,date,columns=["symbol"])
        if df is None or pv is None: return None
        df = self.ensure_et_index(df, time_col="time", tz=ctx.tz)

        out={}
        for sym,g in df.groupby("symbol", observed=True):
            try:
                r = _logret(_rth(g)["close"]).dropna().values
                if r.size == 0: out[sym]=float("nan"); continue
                up = float(np.sum(np.maximum(r, 0.0)))
                dn = float(np.sum(np.maximum(-r, 0.0)))
                out[sym] = float(up/dn) if np.isfinite(dn) and dn>0 else float("nan")
            except Exception:
                out[sym]=float("nan")
        res = pv[["symbol"]].copy(); res["value"] = res["symbol"].map(out); return res

feature = RetUpDownAbsMagnitudeRatio()
