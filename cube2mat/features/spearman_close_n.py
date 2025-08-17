# features/spearman_close_n.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class SpearmanCloseNFeature(BaseFeature):
    """
    09:30–15:59 内，Spearman 等级相关：corr(rank(close), rank(n))。
    若有效样本<2 或秩方差为 0，则 NaN。
    """

    name = "spearman_close_n"
    description = "Spearman rank correlation between close and n within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "n")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
        xd = x - x.mean()
        yd = y - y.mean()
        sxx = (xd * xd).sum()
        syy = (yd * yd).sum()
        if sxx <= 0 or syy <= 0:
            return np.nan
        return float((xd * yd).sum() / np.sqrt(sxx * syy))

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        out = sample[["symbol"]].copy() if sample is not None else None
        if full is None or sample is None:
            return None
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["close", "n"])
        if df.empty:
            out["value"] = pd.NA
            return out

        def per_symbol(g: pd.DataFrame) -> float:
            if len(g) < 2:
                return np.nan
            rc = g["close"].rank(method="average")
            rn = g["n"].rank(method="average")
            return self._pearson_corr(rc, rn)

        value = df.sort_index().groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = SpearmanCloseNFeature()
