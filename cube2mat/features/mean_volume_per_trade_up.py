# features/mean_volume_per_trade_up.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class MeanVolPerTradeUpFeature(BaseFeature):
    """
    上涨状态的“每笔平均量”： sum(volume|ret>0)/sum(n|ret>0)。
    sum(n)<=0 则 NaN。
    """

    name = "mean_volume_per_trade_up"
    description = "Average volume per trade on up bars."
    required_full_columns = ("symbol", "time", "close", "volume", "n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close", "volume", "n"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(sample["symbol"].unique())]

        for c in ("close", "volume", "n"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "volume", "n"]).sort_index()
        if df.empty:
            out["value"] = pd.NA
            return out

        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret"])

        def per_symbol(g: pd.DataFrame) -> float:
            gv = g.loc[g["ret"] > 0, ["volume", "n"]]
            v = float(gv["volume"].sum())
            k = float(gv["n"].sum())
            if not np.isfinite(k) or k <= 0:
                return np.nan
            return float(v / k)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = MeanVolPerTradeUpFeature()
