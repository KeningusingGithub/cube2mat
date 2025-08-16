# features/diff_close_vwap_trend_slope.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class DiffCloseVwapTrendSlopeFeature(BaseFeature):
    """
    09:30–15:59 内，回归 (close - vwap) ~ time(分钟)，输出斜率。
    捕捉 close 对 VWAP 的“溢价/贴水”日内漂移。若样本<2 或 var(time)=0，则 NaN。
    """
    name = "diff_close_vwap_trend_slope"
    description = "OLS slope of (close - vwap) on minutes-since-09:30 within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _ols_slope(x: pd.Series, y: pd.Series) -> float:
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
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None

        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(df_full, "time", ctx.tz).between_time("09:30", "15:59")
        if df is None or df.empty or sample.empty:
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
        tdelta = (df.index - df.index.normalize()) - pd.Timedelta("09:30:00")
        df["t_min"] = tdelta.total_seconds() / 60.0
        df["diff"]  = df["close"] - df["vwap"]

        value = df.groupby("symbol").apply(lambda g: self._ols_slope(g["t_min"], g["diff"]))
        out["value"] = out["symbol"].map(value)
        return out

feature = DiffCloseVwapTrendSlopeFeature()
