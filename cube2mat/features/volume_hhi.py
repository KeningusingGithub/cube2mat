# features/volume_hhi.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VolumeHHIFeature(BaseFeature):
    """
    09:30–15:59 内，成交量 HHI = sum_i ( (vol_i / sum_j vol_j)^2 )；衡量量在时间上的集中度。
    若 sum(vol)<=0 或无有效样本，NaN。
    """

    name = "volume_hhi"
    description = "Herfindahl index of volume distribution across intraday bars."
    required_full_columns = ("symbol", "time", "volume")
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

        df = df.copy()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        def per_symbol(g: pd.DataFrame) -> float:
            tot = g["volume"].sum()
            if not np.isfinite(tot) or tot <= 0:
                return np.nan
            p = g["volume"] / tot
            return float((p * p).sum())

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = VolumeHHIFeature()
