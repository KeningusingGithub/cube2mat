# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df): return df.between_time("09:30", "15:59")

class VolumeMedianToMeanRatio(BaseFeature):
    name = "volume_median_to_mean_ratio"
    description = "Median(volume)/Mean(volume) within RTH; concentration/robustness proxy for volume distribution."
    required_full_columns = ("symbol", "time", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=list(self.required_full_columns))
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, time_col="time", tz=ctx.tz)

        out={}
        for sym, g in df.groupby("symbol", observed=True):
            try:
                v = _rth(g)["volume"].astype(float).dropna()
                m = float(v.mean())
                out[sym] = float(np.median(v.values)/m) if np.isfinite(m) and m>0 else float("nan")
            except Exception:
                out[sym] = float("nan")

        res = pv[["symbol"]].copy(); res["value"] = res["symbol"].map(out); return res

feature = VolumeMedianToMeanRatio()
