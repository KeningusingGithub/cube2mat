# features/parkinson_vol.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class ParkinsonVolFeature(BaseFeature):
    """
    09:30–15:59 内，汇总单日极值：H=max(high), L=min(low)，
    Parkinson 方差：sigma2 = [ln(H/L)]^2 / (4*ln(2))；波动率 = sqrt(sigma2)。
    需 H>0, L>0；若缺列或无效则 NaN。
    """

    name = "parkinson_vol"
    description = "Parkinson volatility from session high/low: sqrt((ln(H/L))^2 / (4 ln 2))."
    required_full_columns = ("symbol", "time", "high", "low")
    required_pv_columns = ("symbol",)

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

        for c in ("high", "low"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["high", "low"])
        if df.empty:
            out["value"] = pd.NA
            return out

        g = df.groupby("symbol")
        H = g["high"].max()
        L = g["low"].min()
        valid = (H > 0) & (L > 0)
        ratio = (H / L).where(valid)
        lnHL = np.log(ratio)
        sigma2 = (lnHL * lnHL) / (4.0 * np.log(2.0))
        value = sigma2.apply(lambda x: float(np.sqrt(x)) if pd.notna(x) and x >= 0 else np.nan)

        out["value"] = out["symbol"].map(value)
        return out


feature = ParkinsonVolFeature()

