# features/trend_slope_volume_weighted.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class TrendSlopeVolumeWeightedFeature(BaseFeature):
    """
    09:30–15:59 内，对每个 symbol 做体量加权回归（WLS）：
        close ~ time(分钟)，权重 = volume。
    输出加权斜率；若有效样本<2、权重和<=0 或加权 var(time)=0，则 NaN。
    """
    name = "trend_slope_volume_weighted"
    description = "Volume-weighted OLS slope of close~time (minutes since 09:30) within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _wls_slope(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
        # 清理非正权重
        w = w.clip(lower=0)
        if (w > 0).sum() < 2:
            return np.nan
        W = w.sum()
        if not np.isfinite(W) or W <= 0:
            return np.nan
        xw = (w * x).sum() / W
        yw = (w * y).sum() / W
        xd = x - xw
        yd = y - yw
        den = (w * (xd * xd)).sum()
        if not np.isfinite(den) or den <= 0:
            return np.nan
        num = (w * (xd * yd)).sum()
        return float(num / den)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df_full, time_col="time", tz=ctx.tz).between_time("09:30", "15:59")
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.copy()
        df["close"]  = pd.to_numeric(df["close"],  errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        tdelta = (df.index - df.index.normalize()) - pd.Timedelta("09:30:00")
        df["t_min"] = tdelta.total_seconds() / 60.0

        value = df.groupby("symbol").apply(
            lambda g: self._wls_slope(g["t_min"], g["close"], g["volume"])
        )
        out["value"] = out["symbol"].map(value)
        return out

feature = TrendSlopeVolumeWeightedFeature()
