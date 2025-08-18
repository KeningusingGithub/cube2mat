from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class AbsRetPerVolume(BaseFeature):
    name = "absret_per_volume"
    description = "Σ|r| (log-returns) divided by total volume in RTH."
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
            r = _logret(gg["close"]).dropna().values
            totv = float(gg["volume"].sum())
            out[sym] = float(np.sum(np.abs(r)) / totv) if r.size > 0 and totv > 0 and np.isfinite(totv) else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = AbsRetPerVolume()
