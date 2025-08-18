# features/share_in_runs_len_ge3.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _run_segments(sgn: np.ndarray):
    s = pd.Series(sgn).astype(int).replace(0, np.nan).dropna().astype(int)
    if s.empty:
        return []
    starts = (s != s.shift()).cumsum()
    groups = s.groupby(starts)
    return [(int(k), len(v)) for k, v in groups]


class ShareInRunsLenGE3Feature(BaseFeature):
    name = "share_in_runs_len_ge3"
    description = "Share of return observations that belong to sign runs with length ≥ 3."
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
            sgn = np.sign(_logret(_rth(g)["close"]).dropna().values)
            segs = _run_segments(sgn)
            total = len(sgn)
            if total == 0:
                out[sym] = float("nan")
                continue
            big = sum(L for _, L in segs if L >= 3)
            out[sym] = float(big / total)
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = ShareInRunsLenGE3Feature()
