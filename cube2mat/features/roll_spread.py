# features/roll_spread.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class RollSpreadFeature(BaseFeature):
    """
    09:30–15:59 内，Roll 有效价差估计：
      Δp_t = close_t - close_{t-1}
      gamma = Cov(Δp_t, Δp_{t-1})（样本协方差）
      若 gamma < 0，spread = 2 * sqrt(-gamma)，否则 NaN。
    使用 bar 级价格近似成交价。样本<3 或无效时 NaN。
    """
    name = "roll_spread"
    description = "Roll effective spread estimator from lag-1 autocovariance of price changes within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _roll_spread_from_series(p: pd.Series) -> float:
        # p 按时间升序
        dp = p.diff().dropna()
        if len(dp) < 2:
            return np.nan
        x = dp.iloc[1:].astype(float)
        y = dp.iloc[:-1].astype(float)
        # 样本协方差
        n = len(x)
        if n < 2:
            return np.nan
        xm = x.mean(); ym = y.mean()
        cov = ((x - xm) * (y - ym)).sum() / (n - 1)
        if np.isfinite(cov) and cov < 0:
            val = 2.0 * np.sqrt(-cov)
            return float(val)
        return np.nan

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
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        value = df.groupby("symbol")["close"].apply(self._roll_spread_from_series)
        out["value"] = out["symbol"].map(value)
        return out

feature = RollSpreadFeature()
