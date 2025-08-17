# features/partial_corr_close_volume_time.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class PartialCorrCloseVolumeTimeFeature(BaseFeature):
    """
    09:30–15:59 内，控制时间(分钟)后的偏相关：
      将 close 与 volume 分别对 time(分钟)做 OLS 回归，取残差，
      然后计算 Pearson corr(resid_close, resid_volume)。
    n<3 或去趋势方差为 0 时 NaN。
    """

    name = "partial_corr_close_volume_time"
    description = "Partial correlation between close and volume controlling for time within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _residual_against_time(z: pd.Series, t: pd.Series) -> pd.Series | None:
        n = len(z)
        if n < 2:
            return None
        zm = z.mean()
        tm = t.mean()
        zd = z - zm
        td = t - tm
        sxx = (td * td).sum()
        if sxx <= 0:
            return None
        beta1 = (td * zd).sum() / sxx
        beta0 = zm - beta1 * tm
        return z - (beta0 + beta1 * t)

    @staticmethod
    def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
        xd = x - x.mean()
        yd = y - y.mean()
        sxx = (xd * xd).sum()
        syy = (yd * yd).sum()
        if sxx <= 0 or syy <= 0:
            return np.nan
        return float((xd * yd).sum() / np.sqrt(sxx * syy))

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
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        tdelta = (df.index - df.index.normalize()) - pd.Timedelta("09:30:00")
        df["t_min"] = tdelta.total_seconds() / 60.0

        def per_symbol(g: pd.DataFrame) -> float:
            rc = self._residual_against_time(g["close"], g["t_min"])
            rv = self._residual_against_time(g["volume"], g["t_min"])
            if rc is None or rv is None:
                return np.nan
            return self._pearson_corr(rc, rv)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = PartialCorrCloseVolumeTimeFeature()
