# features/absret_sqrtvol_slope.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class AbsRetSqrtVolSlopeFeature(BaseFeature):
    """
    09:30–15:59 内，对每个 symbol 回归：
      |ret_t| ~ sqrt(volume_t)
    其中 ret = close.pct_change()；过滤无效/非正 volume。输出斜率；若样本<2 或 var(sqrt(volume))=0 则 NaN。
    """
    name = "absret_sqrtvol_slope"
    description = "OLS slope of |ret| on sqrt(volume) within 09:30–15:59; NaN if insufficient."
    required_full_columns = ("symbol", "time", "close", "volume")
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
        for c in ("close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change()
        df["ret"] = df["ret"].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret"])

        # 对齐 volume 与 ret（首个 ret 缺失已剔除）
        df = df[df["volume"] > 0]
        if df.empty:
            out["value"] = pd.NA
            return out

        df["x"] = np.sqrt(df["volume"])
        df["y"] = df["ret"].abs()

        value = df.groupby("symbol").apply(lambda g: self._ols_slope(g["x"], g["y"]))
        out["value"] = out["symbol"].map(value)
        return out

feature = AbsRetSqrtVolSlopeFeature()
