# features/close_15m_absret_share.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class Close15mAbsRetShareFeature(BaseFeature):
    """
    09:30–15:59 内，收盘 15 分钟的绝对收益贡献占比。
    """

    name = "close_15m_absret_share"
    description = "Share of absolute returns in last 15 minutes; ret by close.pct_change."
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
        g = df.groupby("symbol")
        denom = g["ret"].apply(lambda s: s.abs().sum())
        num = g.apply(
            lambda x: x.loc[x["t_min"] >= (self.TOTAL_MIN - 15.0), "ret"].abs().sum()
        )
        share = (num / denom).where(denom > 0)

        out["value"] = out["symbol"].map(share)
        return out


feature = Close15mAbsRetShareFeature()
