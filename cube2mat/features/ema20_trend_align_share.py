# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df): return df.between_time("09:30","15:59")

class EMA20TrendAlignShare(BaseFeature):
    name = "ema20_trend_align_share"
    description = "Share of RTH bars where sign(Δclose) == sign(ΔEMA20(close)); zeros ignored."
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
                s = _rth(g)["close"].astype(float).dropna()
                if s.size < 3: out[sym]=float("nan"); continue
                ema = s.ewm(span=20, adjust=False).mean()
                sc = np.sign(s.diff())
                se = np.sign(ema.diff())
                mask = sc.ne(0) & se.ne(0) & sc.notna() & se.notna()
                if mask.sum() == 0: out[sym]=float("nan"); continue
                out[sym] = float((sc[mask] == se[mask]).mean())
            except Exception:
                out[sym]=float("nan")

        res = pv[["symbol"]].copy(); res["value"] = res["symbol"].map(out); return res

feature = EMA20TrendAlignShare()
