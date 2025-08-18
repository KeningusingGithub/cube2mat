# features/absret_compaction_80pct_bars_frac.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class AbsRetCompaction80PctBarsFracFeature(BaseFeature):
    name = "absret_compaction_80pct_bars_frac"
    description = (
        "Fraction of RTH bars needed to accumulate 80% of Σ|r| (log-returns), sorting |r| descending."
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
            ar = _logret(_rth(g)["close"]).dropna().abs().values
            n = ar.size
            if n == 0:
                out[sym] = float("nan")
                continue
            tot = float(ar.sum())
            if tot <= 0 or not np.isfinite(tot):
                out[sym] = float("nan")
                continue
            srt = np.sort(ar)[::-1]
            cs = np.cumsum(srt)
            k = int(np.searchsorted(cs, 0.8 * tot, side="left")) + 1
            out[sym] = float(k / n)
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = AbsRetCompaction80PctBarsFracFeature()
