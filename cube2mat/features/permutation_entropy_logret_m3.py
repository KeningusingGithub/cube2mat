from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd, math
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _perm_entropy(x: np.ndarray, m: int = 3) -> float:
    n = x.size
    if n < m:
        return float("nan")
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n - m + 1):
        w = x[i : i + m]
        order = tuple(np.argsort(np.argsort(w, kind="mergesort"), kind="mergesort"))
        counts[order] = counts.get(order, 0) + 1
    p = np.array(list(counts.values()), dtype=float)
    p = p / p.sum()
    H = -np.sum(p * np.log2(p))
    return float(H / np.log2(math.factorial(m)))


class PermutationEntropyLogRetM3(BaseFeature):
    name = "permutation_entropy_logret_m3"
    description = "Permutation entropy (m=3) of RTH log returns, normalized to [0,1] (ordinal patterns)."
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
            out[sym] = _perm_entropy(r, m=3)
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = PermutationEntropyLogRetM3()
