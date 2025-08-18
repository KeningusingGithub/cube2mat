# features/run_balance_index.py
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
    if positive:
        mask = (s == 1).astype(int)
    else:
        mask = (s == -1).astype(int)
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


class RunBalanceIndexFeature(BaseFeature):
    name = "run_balance_index"
    description = (
        "Run-balance index: (mean length of up runs − mean length of down runs) / (sum of the two means)."
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
            sgn = np.sign(_logret(_rth(g)["close"]).dropna().values)
            ups = _run_lengths(pd.Series(sgn), positive=True)
            dns = _run_lengths(pd.Series(sgn), positive=False)
            if len(ups) == 0 or len(dns) == 0:
                out[sym] = float("nan")
                continue
            mu_u = float(np.mean(ups))
            mu_d = float(np.mean(dns))
            denom = mu_u + mu_d
            out[sym] = float((mu_u - mu_d) / denom) if denom > 0 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = RunBalanceIndexFeature()
