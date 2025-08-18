# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")

def _katz_fd(x: pd.Series) -> float:
    s = pd.Series(x).astype(float).dropna()
    n = s.size
    if n < 3:
        return float("nan")
    L = float(np.abs(np.diff(s.values)).sum())
    if not np.isfinite(L) or L <= 0:
        return float("nan")
    d = float(np.max(np.abs(s.values - s.values[0])))
    if not np.isfinite(d) or d <= 0:
        return float("nan")
    a = L / (n - 1.0)
    return float(np.log10(n) / (np.log10(n) + np.log10(d / a)))

class KatzFDClose(BaseFeature):
    name = "katz_fd_close"
    description = "Katz fractal dimension of the close path during RTH (09:30–15:59); higher = more tortuous."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=["symbol"])
        if df_full is None or sample is None:
            return None
        df_full = self.ensure_et_index(df_full, time_col="time", tz=ctx.tz)

        out = {}
        for sym, g in df_full.groupby("symbol", observed=True):
            try:
                val = _katz_fd(_rth(g)["close"])
            except Exception:
                val = float("nan")
            out[sym] = val

        res = sample[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res

feature = KatzFDClose()
