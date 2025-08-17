# features/volume_share_on_up.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VolumeShareOnUpFeature(BaseFeature):
    """
    上涨状态的成交量占比： sum(volume|ret>0) / sum(volume_all)。
    若总量<=0 则 NaN。
    """

    name = "volume_share_on_up"
    description = "Share of total volume that occurs on up bars."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close", "volume"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59").sort_index()
        df = df[df["symbol"].isin(sample["symbol"].unique())]

        for c in ("close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret", "volume"])

        def per_symbol(g: pd.DataFrame) -> float:
            tot = g["volume"].sum()
            if not np.isfinite(tot) or tot <= 0:
                return np.nan
            upv = g.loc[g["ret"] > 0, "volume"].sum()
            return float(upv / tot)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = VolumeShareOnUpFeature()
