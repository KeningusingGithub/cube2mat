from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


class VWAPPositionInRange(BaseFeature):
    name = "vwap_position_in_range"
    description = "Position of session VWAP in the day's H-L range: (VWAP - L)/(H - L)."
    required_full_columns = ("symbol", "time", "high", "low", "vwap", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=list(self.required_full_columns))
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            d = _rth(g)[["high", "low", "vwap", "volume"]].dropna()
            if d.empty:
                out[sym] = float("nan")
                continue
            H, L = float(d["high"].max()), float(d["low"].min())
            rng = H - L
            if not (np.isfinite(rng) and rng > 0):
                out[sym] = float("nan")
                continue
            totv = float(d["volume"].sum())
            if not (np.isfinite(totv) and totv > 0):
                out[sym] = float("nan")
                continue
            vwap_sess = float((d["vwap"] * d["volume"]).sum() / totv)
            out[sym] = float((vwap_sess - L) / rng)
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = VWAPPositionInRange()
