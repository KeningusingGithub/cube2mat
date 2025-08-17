# features/early_late_vol_ratio_30m.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class EarlyLateVolRatio30mFeature(BaseFeature):
    """
    前 30 分钟 vs 后 30 分钟的收益波动比：
      ratio = std(ret in t<30) / std(ret in t>=total_min-30)，ret=close.pct_change()。
    两端样本的有效收益数均需≥3，否则 NaN。
    """

    name = "early_late_vol_ratio_30m"
    description = "Std(ret) first 30m vs last 30m ratio."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

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
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change()
        df["ret"] = df["ret"].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret"])
        if df.empty:
            out["value"] = pd.NA
            return out

        tmin = (
            df.index - df.index.normalize() - pd.Timedelta("09:30:00")
        ).total_seconds() / 60.0
        df["t_min"] = tmin

        def per_symbol(g: pd.DataFrame) -> float:
            early = g.loc[g["t_min"] < 30, "ret"]
            late = g.loc[g["t_min"] >= (self.TOTAL_MIN - 30.0), "ret"]
            n1, n2 = early.count(), late.count()
            if n1 < 3 or n2 < 3:
                return np.nan
            s1 = early.std(ddof=1)
            s2 = late.std(ddof=1)
            if not np.isfinite(s1) or not np.isfinite(s2) or s2 == 0:
                return np.nan
            return float(s1 / s2)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = EarlyLateVolRatio30mFeature()
