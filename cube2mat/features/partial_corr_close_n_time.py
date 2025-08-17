# features/partial_corr_close_n_time.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class PartialCorrCloseNTimeFeature(BaseFeature):
    """
    09:30–15:59 内，控制时间(分钟)后的偏相关：
      resid_close = close ~ time 的残差
      resid_n     = n     ~ time 的残差
      value = corr(resid_close, resid_n)
    若无足够样本或去趋势方差为 0，则 NaN。
    """

    name = "partial_corr_close_n_time"
    description = "Partial correlation between close and n controlling for time within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "n")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _residual_against_time(z: pd.Series, t: pd.Series) -> pd.Series | None:
        if len(z) < 2:
            return None
        zm, tm = z.mean(), t.mean()
        zd, td = z - zm, t - tm
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
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["close", "n"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        tdelta = (df.index - df.index.normalize()) - pd.Timedelta("09:30:00")
        df["t_min"] = tdelta.total_seconds() / 60.0

        def per_symbol(g: pd.DataFrame) -> float:
            rc = self._residual_against_time(g["close"], g["t_min"])
            rn = self._residual_against_time(g["n"], g["t_min"])
            if rc is None or rn is None:
                return np.nan
            return self._pearson_corr(rc, rn)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = PartialCorrCloseNTimeFeature()
