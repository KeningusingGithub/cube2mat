# features/ulcer_index_close.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class UlcerIndexCloseFeature(BaseFeature):
    """Ulcer Index (RMS of drawdowns) for close within 09:30–15:59."""

    name = "ulcer_index_close"
    description = "Ulcer Index based on close drawdowns in RTH (fraction, not %)."
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

        df = (
            self.ensure_et_index(full, "time", ctx.tz)
            .between_time("09:30", "15:59")
        )
        df = df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty:
            out["value"] = pd.NA
            return out
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            s = g.sort_index()["close"]
            if (s <= 0).any() or len(s) < 2:
                res[sym] = np.nan
                continue
            roll_max = s.cummax()
            dd = 1.0 - (s / roll_max)
            dd = dd.clip(lower=0).replace([np.inf, -np.inf], np.nan).dropna()
            res[sym] = float(np.sqrt((dd * dd).mean())) if len(dd) >= 1 else np.nan
        out["value"] = out["symbol"].map(res)
        return out


feature = UlcerIndexCloseFeature()
