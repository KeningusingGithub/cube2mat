from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class RVPerTrade(BaseFeature):
    name = "rv_per_trade"
    description = "Realized variance Σ r^2 divided by total trade count Σ n over RTH."
    required_full_columns = ("symbol", "time", "close", "n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=["symbol", "time", "close", "n"])
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            gg = _rth(g)[["close", "n"]].dropna()
            r = _logret(gg["close"]).dropna().values
            totn = float(gg["n"].sum())
            out[sym] = float(np.sum(r ** 2) / totn) if r.size > 0 and totn > 0 and np.isfinite(totn) else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = RVPerTrade()
