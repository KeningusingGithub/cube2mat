# features/close_15m_volume_share.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class Close15mVolumeShareFeature(BaseFeature):
    """
    09:30–15:59 内，收盘前 15 分钟 (t_min >= total_minutes-15) 的成交量占比。
    若全日有效 volume 总和<=0，则 NaN。
    """

    name = "close_15m_volume_share"
    description = "Share of volume in last 15 minutes."
    required_full_columns = ("symbol", "time", "volume")
    required_pv_columns = ("symbol",)

    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0  # = 389

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

        df = df.sort_index()
        tmin = (
            df.index - df.index.normalize() - pd.Timedelta("09:30:00")
        ).total_seconds() / 60.0
        df["t_min"] = tmin

        g = df.groupby("symbol")
        total = g["volume"].sum()
        close15 = g.apply(
            lambda x: x.loc[x["t_min"] >= (self.TOTAL_MIN - 15.0), "volume"].sum()
        )
        share = (close15 / total).where(total > 0)

        out["value"] = out["symbol"].map(share)
        return out


feature = Close15mVolumeShareFeature()
