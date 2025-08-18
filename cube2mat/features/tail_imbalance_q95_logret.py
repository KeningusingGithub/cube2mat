# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df): return df.between_time("09:30","15:59")
def _logret(s): return np.log(s.astype(float)).diff()

class TailImbalanceQ95LogRet(BaseFeature):
    name = "tail_imbalance_q95_logret"
    description = "Tail imbalance at 95%% of |r|: (count[r≤−t] − count[r≥+t]) / (count[r≤−t] + count[r≥+t])."
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
                r = _logret(_rth(g)["close"]).dropna()
                if r.empty: out[sym]=float("nan"); continue
                t = float(np.nanquantile(np.abs(r.values), 0.95))
                neg = int((r.values <= -t).sum()); pos = int((r.values >= +t).sum())
                denom = neg + pos
                out[sym] = float((neg - pos)/denom) if denom>0 else float("nan")
            except Exception:
                out[sym]=float("nan")
        res = pv[["symbol"]].copy(); res["value"] = res["symbol"].map(out); return res

feature = TailImbalanceQ95LogRet()
