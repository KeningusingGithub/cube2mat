from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


class MidRangeCrossCount(BaseFeature):
    name = "midrange_cross_count"
    description = "Count of sign flips of (close − midrange) where midrange = (H+L)/2 using RTH session H/L; zeros ignored."
    required_full_columns = ("symbol", "time", "high", "low", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=list(self.required_full_columns))
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            d = _rth(g)[["high", "low", "close"]].dropna()
            if d.empty:
                out[sym] = float("nan")
                continue
            H, L = float(d["high"].max()), float(d["low"].min())
            mid = (H + L) / 2.0
            x = d["close"].astype(float) - mid
            s = np.sign(x.values)
            s = s[s != 0]
            out[sym] = float((s[1:] != s[:-1]).sum()) if s.size >= 2 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = MidRangeCrossCount()
