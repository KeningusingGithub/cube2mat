# features/early_vs_late_run_count_ratio.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class EarlyVsLateRunCountRatioFeature(BaseFeature):
    name = "early_vs_late_run_count_ratio"
    description = (
        "Run count in 09:30–10:29 divided by run count in 15:00–15:59 (sign runs of log-returns; zeros break runs)."
    )
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def _run_count_in_window(self, g: pd.DataFrame, start: str, end: str) -> float:
        z = g.between_time(start, end)["close"].astype(float).dropna()
        r = _logret(z).dropna()
        if r.empty:
            return np.nan
        s = np.sign(r.values)
        s = pd.Series(s).replace(0, np.nan).dropna()
        if s.empty:
            return np.nan
        return float((s != s.shift()).sum())

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=["symbol", "time", "close"])
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            early = self._run_count_in_window(g, "09:30", "10:29")
            late = self._run_count_in_window(g, "15:00", "15:59")
            out[sym] = float(early / late) if np.isfinite(late) and late > 0 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = EarlyVsLateRunCountRatioFeature()
