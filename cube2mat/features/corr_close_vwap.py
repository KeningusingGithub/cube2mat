# features/corr_close_vwap.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class CorrCloseVWAPFeature(BaseFeature):
    """
    09:30–15:59 内，close 与 vwap 的 Pearson 相关。
    样本 <3 或任一方差为 0 时 NaN。
    """

    name = "corr_close_vwap"
    description = "Correlation of close and vwap within RTH."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = min(x.size, y.size)
        if n < 3:
            return np.nan
        xc = x - x.mean()
        yc = y - y.mean()
        sx = np.sqrt((xc * xc).sum())
        sy = np.sqrt((yc * yc).sum())
        if sx <= 0 or sy <= 0 or not np.isfinite(sx * sy):
            return np.nan
        return float((xc * yc).sum() / (sx * sy))

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
        df = df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["vwap"] = pd.to_numeric(df["vwap"], errors="coerce")
        df = df.dropna(subset=["close", "vwap"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res = df.groupby("symbol").apply(lambda g: self._corr(g.sort_index()["close"].to_numpy(), g.sort_index()["vwap"].to_numpy()))
        out["value"] = out["symbol"].map(res)
        return out


feature = CorrCloseVWAPFeature()
