# features/ar1_ret_coef.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class AR1RetCoefFeature(BaseFeature):
    """
    09:30–15:59 内，计算 ret_t = α + φ * ret_{t-1} + ε 的 OLS φ（AR(1) 系数）。
    其中 ret = close.pct_change()，清理 inf/NaN。若有效样本对 < 3（即 ret 数≥4）或 var(ret_{t-1})=0，则 NaN。
    """
    name = "ar1_ret_coef"
    description = "AR(1) coefficient φ for intraday simple returns ret=close.pct_change(), within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _ols_beta(x: pd.Series, y: pd.Series) -> float:
        n = len(x)
        if n < 3:
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
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        # 计算 ret
        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change()
        df["ret"] = df["ret"].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret"])

        if df.empty:
            out["value"] = pd.NA
            return out

        # 构造滞后对
        df["ret_lag1"] = df.groupby("symbol", sort=False)["ret"].shift(1)
        df = df.dropna(subset=["ret", "ret_lag1"])

        if df.empty:
            out["value"] = pd.NA
            return out

        value = df.groupby("symbol").apply(
            lambda g: self._ols_beta(g["ret_lag1"], g["ret"])
        )

        out["value"] = out["symbol"].map(value)
        return out

feature = AR1RetCoefFeature()
