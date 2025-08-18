from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


class CloseToEdgeDistanceFrac(BaseFeature):
    name = "close_to_edge_distance_frac"
    description = "Normalized edge proximity: min(C-L, H-C)/(H-L) using RTH H/L and last close."
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
            H, L, C = float(d["high"].max()), float(d["low"].min()), float(d["close"].iloc[-1])
            rng = H - L
            out[sym] = float(min(C - L, H - C) / rng) if np.isfinite(rng) and rng > 0 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = CloseToEdgeDistanceFrac()
