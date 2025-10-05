from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _apen(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= m + 1:
        return float("nan")
    if r is None:
        r = 0.2 * np.nanstd(x)
        if not np.isfinite(r) or r <= 0:
            return float("nan")
    def _phi(mm: int) -> float:
        if n - mm + 1 <= 0:
            return float("nan")
        X = np.array([x[i : i + mm] for i in range(n - mm + 1)])
        if X.size == 0:
            return float("nan")
        diff = np.abs(X[:, None, :] - X[None, :, :])
        C = (diff.max(axis=2) <= r).mean(axis=1)
        C = C[C > 0]
        return float(np.mean(np.log(C))) if C.size > 0 else float("nan")
    p1 = _phi(m)
    p2 = _phi(m + 1)
    return float(p1 - p2) if np.isfinite(p1) and np.isfinite(p2) else float("nan")


class ApproxEntropyAbsRetM2R02(BaseFeature):
    name = "approx_entropy_absret_m2_r02"
    description = "Approximate Entropy of |log returns| with m=2, r=0.2·std(|r|) within RTH."
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
            out[sym] = _apen(r, m=2, r=0.2 * np.nanstd(r)) if r.size > 5 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"] .map(out)
        return res


feature = ApproxEntropyAbsRetM2R02()
