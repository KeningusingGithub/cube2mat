# features/run_count_total.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _run_count(signs: pd.Series) -> float:
    s = pd.Series(signs).astype(int)
    s = s.replace(0, np.nan).dropna()
    if s.empty:
        return float("nan")
    return float((s != s.shift()).sum())


class RunCountTotalFeature(BaseFeature):
    name = "run_count_total"
    description = "Total number of sign runs in intraday log-returns (zeros break runs)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=["symbol", "time", "close"])
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            r = _logret(_rth(g)["close"]).dropna()
            s = np.sign(r.values)
            out[sym] = _run_count(pd.Series(s))
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = RunCountTotalFeature()
