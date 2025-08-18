# features/return_contribution_top_quintile_by_volume.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class ReturnContributionTopQuintileByVolumeFeature(BaseFeature):
    name = "return_contribution_top_quintile_by_volume"
    description = (
        "Share of Σ|r| contributed by bars in the top 20% of volume (threshold by volume 80th percentile)."
    )
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
            if gg.empty:
                out[sym] = float("nan")
                continue
            r = _logret(gg["close"]).dropna()
            if r.empty:
                out[sym] = float("nan")
                continue
            v = gg["volume"].astype(float).reindex(r.index)
            thr = float(np.nanquantile(v.values, 0.80))
            mask = v.values >= thr
            num = float(np.abs(r.values[mask]).sum())
            den = float(np.abs(r.values).sum())
            out[sym] = (num / den) if den > 0 and np.isfinite(den) else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = ReturnContributionTopQuintileByVolumeFeature()
