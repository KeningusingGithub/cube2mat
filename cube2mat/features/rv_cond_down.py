# features/rv_cond_down.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RVCondDownFeature(BaseFeature):
    """
    仅在下行 bar (simple ret<0) 时累积的 RV：∑ r^2 (logret)。
    无满足条件的收益 → NaN。
    """

    name = "rv_cond_down"
    description = "RV contributed by down bars (simple ret<0) in RTH."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, list(self.required_full_columns))
        sample = self.load_pv(ctx, date, list(self.required_pv_columns))
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        df = df[df.symbol.isin(sample.symbol.unique())].copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            rlog = np.log(g["close"]).diff().replace([np.inf, -np.inf], np.nan)
            rsimple = g["close"].pct_change()
            mask = rsimple < 0
            rr = (rlog * rlog)[mask]
            res[sym] = float(rr.dropna().sum()) if rr.notna().sum() >= 1 else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = RVCondDownFeature()
