from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _sampen(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= m + 1:
        return float("nan")
    if r is None:
        r = 0.2 * np.nanstd(x)
        if not np.isfinite(r) or r <= 0:
            return float("nan")
    def _count(mm: int) -> int:
        cnt = 0
        for i in range(n - mm):
            xi = x[i : i + mm]
            for j in range(i + 1, n - mm + 1):
                xj = x[j : j + mm]
                if np.max(np.abs(xi - xj)) <= r:
                    cnt += 1
        return cnt
    A = _count(m + 1)
    B = _count(m)
    return float(-np.log(A / B)) if A > 0 and B > 0 else float("nan")


class SampleEntropyAbsRetM2R02(BaseFeature):
    name = "sample_entropy_absret_m2_r02"
    description = "Sample Entropy of |log returns| with m=2, r=0.2·std(|r|) within RTH."
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
            r = np.abs(_logret(_rth(g)["close"]).dropna().values)
            out[sym] = _sampen(r, m=2, r=0.2 * np.nanstd(r)) if r.size > 5 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = SampleEntropyAbsRetM2R02()
