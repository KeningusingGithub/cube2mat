# features/volume_top3_share.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


class VolumeTop3ShareFeature(BaseFeature):
    name = "volume_top3_share"
    description = "Share of total RTH volume contributed by the top-3 volume bars."
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
            v = _rth(g)["volume"].astype(float).dropna().values
            if v.size == 0:
                out[sym] = float("nan")
                continue
            tot = float(v.sum())
            if tot <= 0 or not np.isfinite(tot):
                out[sym] = float("nan")
                continue
            top3 = float(np.sort(v)[-3:].sum()) if v.size >= 3 else float(np.sort(v).sum())
            out[sym] = top3 / tot
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = VolumeTop3ShareFeature()
