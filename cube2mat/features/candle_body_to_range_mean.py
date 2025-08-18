# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df): return df.between_time("09:30","15:59")

class CandleBodyToRangeMean(BaseFeature):
    name = "candle_body_to_range_mean"
    description = "Mean of |close-open|/(high-low) across RTH bars (exclude zero-range bars)."
    required_full_columns = ("symbol","time","open","high","low","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx,date,columns=list(self.required_full_columns))
        pv = self.load_pv(ctx,date,columns=["symbol"])
        if df is None or pv is None: return None
        df = self.ensure_et_index(df, time_col="time", tz=ctx.tz)

        out={}
        for sym,g in df.groupby("symbol", observed=True):
            try:
                d = _rth(g)[["open","high","low","close"]].dropna().astype(float)
                rng = (d["high"] - d["low"]).replace(0.0, np.nan)
                ratio = np.abs(d["close"] - d["open"]) / rng
                out[sym] = float(np.nanmean(ratio.values)) if ratio.size>0 else float("nan")
            except Exception:
                out[sym]=float("nan")
        res = pv[["symbol"]].copy(); res["value"] = res["symbol"].map(out); return res

feature = CandleBodyToRangeMean()
