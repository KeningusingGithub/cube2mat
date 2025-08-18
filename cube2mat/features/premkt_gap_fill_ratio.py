# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _premkt(df): return df.between_time("00:00","09:29")
def _rth(df): return df.between_time("09:30","15:59")

class PreMktGapFillRatio(BaseFeature):
    name = "premkt_gap_fill_ratio"
    description = "Gap-fill ratio vs premarket: -(close_RTH_last - open_RTH_first) / (open_RTH_first - premarket_last_close). 1=完全回补，>1=过度回补。"
    required_full_columns = ("symbol","time","open","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx,date,columns=list(self.required_full_columns))
        pv = self.load_pv(ctx,date,columns=["symbol"])
        if df is None or pv is None: return None
        df = self.ensure_et_index(df, time_col="time", tz=ctx.tz)

        out={}
        for sym,g in df.groupby("symbol", observed=True):
            try:
                pre = _premkt(g)[["close"]].dropna()
                rth = _rth(g)[["open","close"]].dropna()
                if pre.empty or rth.empty: out[sym]=float("nan"); continue
                pre_last = float(pre["close"].iloc[-1])
                op = float(rth["open"].iloc[0])
                cl = float(rth["close"].iloc[-1])
                gap = op - pre_last
                if not np.isfinite(gap) or gap == 0:
                    out[sym] = float("nan")
                else:
                    out[sym] = float(-(cl - op)/gap)
            except Exception:
                out[sym]=float("nan")
        res = pv[["symbol"]].copy(); res["value"] = res["symbol"].map(out); return res

feature = PreMktGapFillRatio()
