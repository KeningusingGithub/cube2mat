# features/relprice_log_elasticity_n.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RelPriceLogElasticityNFeature(BaseFeature):
    """
    09:30–15:59 内，弹性回归：log(close/anchor) ~ log(n)，anchor=首笔 open（若无则首笔 close）。
    仅使用 close>0 且 n>0 的样本。返回斜率（对数弹性）。
    若有效样本<2 或 var(log(n))=0，则 NaN。
    """

    name = "relprice_log_elasticity_n"
    description = "Elasticity: slope of log(close/anchor) on log(n) within 09:30–15:59."
    required_full_columns = ("symbol", "time", "open", "close", "n")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _ols_slope(x: pd.Series, y: pd.Series) -> float:
        if len(x) < 2:
            return np.nan
        xd = x - x.mean()
        yd = y - y.mean()
        sxx = (xd * xd).sum()
        if sxx <= 0:
            return np.nan
        return float(((xd * yd).sum()) / sxx)

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
        for c in ("open", "close", "n"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "n"])
        df = df[(df["close"] > 0) & (df["n"] > 0)]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()

        def per_symbol(g: pd.DataFrame) -> float:
            first_open = g["open"].dropna()
            anchor = first_open.iloc[0] if not first_open.empty else g["close"].iloc[0]
            if not np.isfinite(anchor) or anchor <= 0:
                return np.nan
            y = np.log(g["close"]) - np.log(anchor)
            x = np.log(g["n"])
            mask = (
                y.replace([np.inf, -np.inf], np.nan).notna()
                & x.replace([np.inf, -np.inf], np.nan).notna()
            )
            if mask.sum() < 2:
                return np.nan
            return self._ols_slope(x[mask], y[mask])

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = RelPriceLogElasticityNFeature()
