# features/open_15m_absret_share.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class Open15mAbsRetShareFeature(BaseFeature):
    """
    09:30–15:59 内，开盘 15 分钟的绝对收益贡献占比：
      share = sum(|ret| in t<15) / sum(|ret| all), ret=close.pct_change()。
    若全日有效 ret 为空或分母为 0，则 NaN。
    """

    name = "open_15m_absret_share"
    description = "Share of absolute returns in first 15 minutes; ret by close.pct_change."
    required_full_columns = ("symbol", "time", "close")
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
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        # ret
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
        num = g.apply(lambda x: x.loc[x["t_min"] < 15, "ret"].abs().sum())
        share = (num / denom).where(denom > 0)

        out["value"] = out["symbol"].map(share)
        return out


feature = Open15mAbsRetShareFeature()
