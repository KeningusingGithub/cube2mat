# features/bars_to_50pct_volume_time_frac.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


class BarsTo50PctVolumeTimeFracFeature(BaseFeature):
    name = "bars_to_50pct_volume_time_frac"
    description = (
        "In time order, fraction of RTH bars elapsed when cumulative volume first reaches 50% of the day's total."
    )
    required_full_columns = ("symbol", "time", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=["symbol", "time", "volume"])
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            v = _rth(g)["volume"].astype(float).dropna()
            n = v.size
            if n == 0:
                out[sym] = float("nan")
                continue
            tot = float(v.sum())
            if tot <= 0 or not np.isfinite(tot):
                out[sym] = float("nan")
                continue
            cs = v.cumsum().values
            k = int(np.searchsorted(cs, 0.5 * tot, side="left")) + 1
            out[sym] = float(k / n)
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = BarsTo50PctVolumeTimeFracFeature()
