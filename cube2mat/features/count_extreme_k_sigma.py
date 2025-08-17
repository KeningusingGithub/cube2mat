# features/count_extreme_k_sigma.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class CountExtremeKSigmaFeature(BaseFeature):
    """Count of extreme simple returns where |ret| > k * sigma_robust."""

    name = "count_extreme_k_sigma"
    description = "Count of bars with |ret| exceeding k·σ_robust (σ from IQR) within RTH."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)
    k = 3.0

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

        k = self.k
        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            r = (
                g.sort_index()["close"].pct_change()
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(r) < 3:
                res[sym] = np.nan
                continue
            q1, q3 = r.quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            sigma = 0.7413 * iqr if iqr > 0 else float(r.std(ddof=1))
            if not np.isfinite(sigma) or sigma <= 0:
                res[sym] = np.nan
                continue
            cnt = int((r.abs() > (k * sigma)).sum())
            res[sym] = float(cnt)
        out["value"] = out["symbol"].map(res)
        return out


feature = CountExtremeKSigmaFeature()
