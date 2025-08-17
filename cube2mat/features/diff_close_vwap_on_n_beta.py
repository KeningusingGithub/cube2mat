# features/diff_close_vwap_on_n_beta.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class DiffCloseVwapOnNBetaFeature(BaseFeature):
    """
    09:30–15:59 内，OLS: (close - vwap) ~ n，输出斜率。
    捕捉价差溢价与成交笔数的关系。样本<2 或 var(n)=0 则 NaN。
    """

    name = "diff_close_vwap_on_n_beta"
    description = "OLS slope of (close - vwap) on n within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "vwap", "n")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _ols_slope(x: pd.Series, y: pd.Series) -> float:
        if len(x) < 2:
            return np.nan
        xd = x - x.mean()
        yd = y - y.mean()
        sxx = (xd * xd).sum()
        if sxx <= 0:
            return np.nan
        return float(((xd * yd).sum()) / sxx)

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
        for c in ("close", "vwap", "n"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "vwap", "n"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df["diff"] = df["close"] - df["vwap"]
        df = df.sort_index()
        value = df.groupby("symbol").apply(
            lambda g: self._ols_slope(g["n"], g["diff"])
        )
        out["value"] = out["symbol"].map(value)
        return out


feature = DiffCloseVwapOnNBetaFeature()
