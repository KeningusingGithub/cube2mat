# features/run_len_var_ratio_up_to_down.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _run_lengths(sgn: pd.Series, positive: bool = True):
    s = pd.Series(sgn).astype(int)
    s[s == 0] = 0
    s = s.replace({1: 1, -1: -1})
    mask = (s == 1).astype(int) if positive else (s == -1).astype(int)
    if mask.sum() == 0:
        return []
    diff = mask.diff().fillna(0).values
    starts = np.where(diff == 1)[0]
    if mask.iloc[0] == 1:
        starts = np.r_[0, starts]
    ends = np.where(diff == -1)[0] - 1
    if mask.iloc[-1] == 1:
        ends = np.r_[ends, len(mask) - 1]
    return (ends - starts + 1).tolist()


class RunLenVarRatioUpToDownFeature(BaseFeature):
    name = "run_len_var_ratio_up_to_down"
    description = "Variance of up-run lengths divided by variance of down-run lengths (zeros break runs)."
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
            ups = _run_lengths(pd.Series(sgn), True)
            dns = _run_lengths(pd.Series(sgn), False)
            if len(ups) < 2 or len(dns) < 2:
                out[sym] = float("nan")
                continue
            vu = float(np.var(ups, ddof=1))
            vd = float(np.var(dns, ddof=1))
            out[sym] = float(vu / vd) if vd > 0 and np.isfinite(vd) else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = RunLenVarRatioUpToDownFeature()
