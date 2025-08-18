# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df): return df.between_time("09:30","15:59")

class VWAPDevRelMedianAbs(BaseFeature):
    name = "vwap_dev_rel_median_abs"
    description = "Median of |(close - vwap)/vwap| across RTH bars (robust VWAP deviation scale)."
    required_full_columns = ("symbol","time","close","vwap")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx,date,columns=list(self.required_full_columns))
        pv = self.load_pv(ctx,date,columns=["symbol"])
        if df is None or pv is None: return None
        df = self.ensure_et_index(df, time_col="time", tz=ctx.tz)

        out={}
        for sym,g in df.groupby("symbol", observed=True):
            try:
                d = _rth(g)[["close","vwap"]].dropna()
                if d.empty: out[sym]=float("nan"); continue
                dev = (d["close"].astype(float) - d["vwap"].astype(float)) / d["vwap"].astype(float)
                dev = dev.replace([np.inf, -np.inf], np.nan).dropna()
                out[sym] = float(np.median(np.abs(dev.values))) if dev.size>0 else float("nan")
            except Exception:
                out[sym]=float("nan")
        res = pv[["symbol"]].copy(); res["value"] = res["symbol"].map(out); return res

feature = VWAPDevRelMedianAbs()
