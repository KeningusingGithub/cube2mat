from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = pd.Series(a).rank(method="average").values
    b = pd.Series(b).rank(method="average").values
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class SpearmanAbsRetVolume(BaseFeature):
    name = "spearman_absret_volume"
    description = "Spearman rank correlation between |log returns| and volume in RTH."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=["symbol", "time", "close", "volume"])
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            gg = _rth(g)[["close", "volume"]].dropna()
            r = _logret(gg["close"]).dropna()
            if r.empty:
                out[sym] = float("nan")
                continue
            v = gg["volume"].astype(float).reindex(r.index).values
            out[sym] = _spearman(np.abs(r.values), v)
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = SpearmanAbsRetVolume()
