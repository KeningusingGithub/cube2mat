# features/absret_peak_time_min.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class AbsRetPeakTimeMinFeature(BaseFeature):
    """
    |ret| 峰值出现时间占全日交易时长的比例（自 09:30 起），ret=close.pct_change()；若无有效 ret 则 NaN。
    """

    name = "absret_peak_time_min"
    description = "Fraction of trading session elapsed when |ret| reaches its maximum."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

    @staticmethod
    def _minutes_since_open(ts: pd.Timestamp) -> float:
        td = (
            ts - ts.normalize() - pd.Timedelta("09:30:00")
        ).total_seconds() / 60.0
        return float(td)

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
        df["absret"] = df["ret"].abs()
        df = df.dropna(subset=["absret"])
        if df.empty:
            out["value"] = pd.NA
            return out

        def per_symbol(g: pd.DataFrame) -> float:
            idx = g["absret"].idxmax()
            if pd.isna(idx):
                return np.nan
            return self._minutes_since_open(idx) / self.TOTAL_MIN

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = AbsRetPeakTimeMinFeature()
