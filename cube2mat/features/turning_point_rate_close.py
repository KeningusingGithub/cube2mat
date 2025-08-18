# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df): return df.between_time("09:30", "15:59")

class TurningPointRateClose(BaseFeature):
    name = "turning_point_rate_close"
    description = "Fraction of interior RTH bars that are local turning points of close (sign of Δclose flips; endpoints excluded)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=list(self.required_full_columns))
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, time_col="time", tz=ctx.tz)

        out = {}
        for sym, g in df.groupby("symbol", observed=True):
            try:
                s = _rth(g)["close"].astype(float).dropna()
                if s.size < 3:
                    out[sym] = float("nan"); continue
                d = s.diff().to_numpy()
                sign = np.sign(d)
                mid = sign[1:-1]; nxt = sign[2:]
                valid = (mid != 0) & (nxt != 0)
                if mid.size == 0:
                    out[sym] = float("nan"); continue
                tp = ((mid * nxt) < 0) & valid
                out[sym] = float(tp.sum() / (s.size - 2))
            except Exception:
                out[sym] = float("nan")

        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res

feature = TurningPointRateClose()
