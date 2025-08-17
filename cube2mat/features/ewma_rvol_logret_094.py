# features/ewma_rvol_logret_094.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class EWMARVolLogret094Feature(BaseFeature):
    """
    EWMA realized volatility (lambda=0.94) of intraday log returns within RTH.
    Weights normalized to sum to 1:
      w_i = (1 - λ) / (1 - λ^n) * λ^{n-1-i}; sigma = sqrt(sum w_i r_i^2).
    """

    name = "ewma_rvol_logret_094"
    description = "RiskMetrics-style EWMA RV (lambda=0.94) on log returns (RTH)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)
    lam = 0.94

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59")
        df = df[df.symbol.isin(sample.symbol.unique())]
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        l = float(self.lam)
        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            r = (
                np.log(g.sort_index()["close"]).diff().replace([np.inf, -np.inf], np.nan).dropna()
            ).to_numpy()
            n = r.size
            if n < 3:
                res[sym] = np.nan
                continue
            w = (1 - l) * l ** np.arange(n - 1, -1, -1)
            w = w / w.sum()
            sigma = float(np.sqrt(np.sum(w * (r * r))))
            res[sym] = sigma if np.isfinite(sigma) else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = EWMARVolLogret094Feature()
