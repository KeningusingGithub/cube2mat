# features/time_to_first_extreme_q90_absret_frac.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class TimeToFirstExtremeQ90AbsRetFracFeature(BaseFeature):
    name = "time_to_first_extreme_q90_absret_frac"
    description = (
        "Fraction of RTH returns elapsed when |r| first crosses the 90th percentile of |r| (intraday log-returns)."
    )
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
            r = _logret(_rth(g)["close"]).dropna().values
            n = r.size
            if n == 0:
                out[sym] = float("nan")
                continue
            thr = float(np.nanquantile(np.abs(r), 0.90))
            idx = np.where(np.abs(r) >= thr)[0]
            out[sym] = float((idx[0] + 1) / n) if idx.size > 0 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = TimeToFirstExtremeQ90AbsRetFracFeature()
