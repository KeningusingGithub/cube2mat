# features/corr_ret_volume.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class CorrRetVolumeFeature(BaseFeature):
    """
    09:30–15:59 内，logret 与 volume 的 Pearson 相关。
    成对样本 <3 或任一方差为 0 时 NaN。
    """

    name = "corr_ret_volume"
    description = "Correlation of signed log returns and volume within RTH."
    required_full_columns = ("symbol", "time", "close", "volume")
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
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            r = np.log(g["close"]).diff().replace([np.inf, -np.inf], np.nan)
            xy = pd.concat([r, g["volume"]], axis=1).dropna()
            res[sym] = self._corr(xy.iloc[:, 0].to_numpy(), xy.iloc[:, 1].to_numpy())

        out["value"] = out["symbol"].map(res)
        return out


feature = CorrRetVolumeFeature()
