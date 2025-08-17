# features/vwap_dev_ar1_halflife_bars.py
from __future__ import annotations
import datetime as dt
from math import log
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VWAPDevAR1HalfLifeBarsFeature(BaseFeature):
    """
    Half-life (in bars) of deviation d_t = close - vwap under AR(1):
      d_t = c + phi * d_{t-1} + e.
    Half-life = -ln(2)/ln(phi) if 0 < phi < 1; otherwise NaN.
    """

    name = "vwap_dev_ar1_halflife_bars"
    description = "AR(1) half-life of (close - vwap) deviation in RTH (bars)."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "close", "vwap"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59").copy()
        for c in ("close", "vwap"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "vwap"])
        df = df[df.symbol.isin(sample.symbol.unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            d = (g.sort_index()["close"] - g["vwap"]).to_numpy(dtype=float)
            if d.size < 3:
                res[sym] = np.nan
                continue
            y = d[1:]
            x = d[:-1]
            X = np.column_stack([np.ones(x.size), x])
            try:
                beta, _ = np.linalg.lstsq(X, y, rcond=None)
                phi = float(beta[1])
                if 0 < phi < 1:
                    hl = -log(2) / log(phi)
                    res[sym] = float(hl) if np.isfinite(hl) else np.nan
                else:
                    res[sym] = np.nan
            except Exception:
                res[sym] = np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = VWAPDevAR1HalfLifeBarsFeature()
