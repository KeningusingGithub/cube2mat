from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


class PriceAboveMedianShare(BaseFeature):
    name = "price_above_median_share"
    description = "Share of RTH bars with close strictly above the RTH median close."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=list(self.required_full_columns))
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            s = _rth(g)["close"].astype(float).dropna()
            if s.empty:
                out[sym] = float("nan")
                continue
            med = float(np.median(s.values))
            out[sym] = float((s.values > med).mean())
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = PriceAboveMedianShare()
