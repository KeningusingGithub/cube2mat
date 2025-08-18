from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _higuchi_fd(x: np.ndarray, kmax: int = 7) -> float:
    N = x.size
    if N < kmax + 2:
        return float("nan")
    Lk = []
    K = []
    for k in range(2, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if idx.size < 2:
                continue
            y = x[idx]
            L = np.sum(np.abs(np.diff(y))) * (N - 1) / ((idx.size - 1) * k)
            Lm.append(L)
        if len(Lm) == 0:
            continue
        Lk.append(np.mean(Lm))
        K.append(1.0 / k)
    if len(Lk) < 2:
        return float("nan")
    X = np.log(K)
    Y = np.log(Lk)
    slope = np.polyfit(X, Y, 1)[0]
    return float(slope)


class HiguchiFDLogRetKmax7(BaseFeature):
    name = "higuchi_fd_logret_kmax7"
    description = "Higuchi fractal dimension estimate of log returns (k_max=7) within RTH."
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
            out[sym] = _higuchi_fd(r, kmax=7)
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = HiguchiFDLogRetKmax7()
