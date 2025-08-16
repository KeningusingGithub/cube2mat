# features/impact_slope_price_cumvol.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class ImpactSlopePriceCumVolFeature(BaseFeature):
    """
    09:30–15:59 内，回归 y ~ x：
      y = close - first_open   （以首笔 open 为锚的价格变动）
      x = 累积成交量 cumsum(volume)
    输出斜率（单位：价格/股）。若样本<2 或 var(x)=0，则 NaN。
    若首笔 open 缺失，退化为以首笔 close 为锚。
    """
    name = "impact_slope_price_cumvol"
    description = "OLS slope of (close - first_open) on cumulative volume within 09:30–15:59; NaN if insufficient."
    required_full_columns = ("symbol", "time", "open", "close", "volume")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _ols_slope_xy(x: pd.Series, y: pd.Series) -> float:
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

    @staticmethod
    def _first_valid(s: pd.Series):
        s = s.dropna()
        return None if s.empty else float(s.iloc[0])

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
        for c in ("open", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()

        def per_symbol(g: pd.DataFrame) -> float:
            # 锚定价：优先首个有效 open，否则首个 close
            first_open = g["open"].dropna()
            anchor = first_open.iloc[0] if not first_open.empty else g["close"].iloc[0]
            x = g["volume"].cumsum()
            y = g["close"] - anchor
            # 清理
            mask = x.notna() & y.notna()
            x = x[mask]; y = y[mask]
            return ImpactSlopePriceCumVolFeature._ols_slope_xy(x, y)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out

feature = ImpactSlopePriceCumVolFeature()
