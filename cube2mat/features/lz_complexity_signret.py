from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _lz_complexity_binary(seq: str) -> float:
    n = len(seq)
    if n == 0:
        return float("nan")
    i = 0
    c = 0
    while i < n:
        l = 1
        while i + l <= n and seq[i : i + l] in seq[:i]:
            l += 1
        c += 1
        i += l
    norm = (n / np.log2(n)) if n > 1 else 1.0
    return float(c / norm)


class LZComplexitySignRet(BaseFeature):
    name = "lz_complexity_signret"
    description = "Lempel–Ziv complexity of the sign of log returns (zeros dropped), normalized by n/log2(n)."
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
            r = np.sign(_logret(_rth(g)["close"]).dropna().values)
            r = r[r != 0]
            s = "".join("1" if v > 0 else "0" for v in r)
            out[sym] = _lz_complexity_binary(s)
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = LZComplexitySignRet()
