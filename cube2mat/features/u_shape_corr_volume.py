# features/u_shape_corr_volume.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class UShapeCorrVolumeFeature(BaseFeature):
    """Pearson correlation between volume and a U-shape template over RTH bars."""

    name = "u_shape_corr_volume"
    description = "Correlation of volume with U-shape time template (early/late high)."
    required_full_columns = ("symbol", "time", "volume")
    required_pv_columns = ("symbol",)

    def _template(self, n: int) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n, endpoint=True)
        u = (t - 0.5) ** 2
        std = u.std(ddof=1)
        u = (u - u.mean()) / (std if std > 0 else 1.0)
        return u

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
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            v = g.sort_index()["volume"]
            if len(v) < 3:
                res[sym] = np.nan
                continue
            x = v.to_numpy(dtype=float)
            x = x - x.mean()
            sx = np.sqrt((x * x).sum())
            if not np.isfinite(sx) or sx == 0:
                res[sym] = np.nan
                continue
            u = self._template(len(x))
            su = np.sqrt((u * u).sum())
            if su == 0:
                res[sym] = np.nan
                continue
            corr = float(np.dot(x, u) / (sx * su))
            res[sym] = corr
        out["value"] = out["symbol"].map(res)
        return out


feature = UShapeCorrVolumeFeature()
