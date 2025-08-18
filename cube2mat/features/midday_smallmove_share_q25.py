# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df): return df.between_time("09:30","15:59")
def _logret(s): return np.log(s.astype(float)).diff()

class MiddaySmallMoveShareQ25(BaseFeature):
    name = "midday_smallmove_share_q25"
    description = "Share of bars in 11:00–14:00 with |logret| ≤ overall RTH 25th percentile of |logret|."
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
                z = _rth(g)["close"].dropna()
                r = _logret(z).dropna()
                if r.empty: out[sym]=float("nan"); continue
                q25 = float(np.nanquantile(np.abs(r.values), 0.25))
                mid_idx = _rth(g).between_time("11:00","14:00").index
                rm = r.loc[r.index.intersection(mid_idx)]
                out[sym] = float((np.abs(rm.values) <= q25).mean()) if rm.size>0 else float("nan")
            except Exception:
                out[sym]=float("nan")
        res = pv[["symbol"]].copy(); res["value"] = res["symbol"].map(out); return res

feature = MiddaySmallMoveShareQ25()
