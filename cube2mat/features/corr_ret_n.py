# features/corr_ret_n.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class CorrRetNFeature(BaseFeature):
    """
    Pearson correlation between signed log returns and trade count n in RTH.
    """

    name = "corr_ret_n"
    description = "Correlation of log returns and trade count n (RTH)."
    required_full_columns = ("symbol", "time", "close", "n")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size != y.size or x.size < 3:
            return np.nan
        xc = x - x.mean()
        yc = y - y.mean()
        sx = float(np.sqrt(np.sum(xc * xc)))
        sy = float(np.sqrt(np.sum(yc * yc)))
        if sx <= 0 or sy <= 0 or not np.isfinite(sx * sy):
            return np.nan
        return float(np.sum(xc * yc) / (sx * sy))

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "close", "n"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59").copy()
        for c in ("close", "n"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "n"])
        df = df[df.symbol.isin(sample.symbol.unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            r = np.log(g["close"]).diff().replace([np.inf, -np.inf], np.nan)
            xy = pd.concat([r, g["n"]], axis=1).dropna()
            res[sym] = self._corr(xy.iloc[:, 0].to_numpy(), xy.iloc[:, 1].to_numpy())

        out["value"] = out["symbol"].map(res)
        return out


feature = CorrRetNFeature()
