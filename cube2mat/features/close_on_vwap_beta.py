# features/close_on_vwap_beta.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class CloseOnVwapBetaFeature(BaseFeature):
    """
    09:30–15:59 内，回归 close ~ vwap，输出斜率（beta）。
    用于刻画 close 对“交易加权锚(vwap)”的弹性；若样本<2 或 var(vwap)=0，则 NaN。
    """
    name = "close_on_vwap_beta"
    description = "OLS slope (beta) of close on vwap within 09:30–15:59; NaN if insufficient."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _beta(x: pd.Series, y: pd.Series) -> float:
        n = len(x)
        if n < 2:
            return np.nan
        xm = x.mean(); ym = y.mean()
        xd = x - xm; yd = y - ym
        den = (xd * xd).sum()
        if den <= 0:
            return np.nan
        num = (xd * yd).sum()
        return float(num / den)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if full is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.copy()
        for c in ("close", "vwap"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "vwap"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        value = df.groupby("symbol").apply(lambda g: self._beta(g["vwap"], g["close"]))

        out["value"] = out["symbol"].map(value)
        return out

feature = CloseOnVwapBetaFeature()
