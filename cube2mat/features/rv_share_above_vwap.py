# features/rv_share_above_vwap.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RVShareAboveVWAPFeature(BaseFeature):
    """
    以 log 收益 r=diff(log(close)) 计算 RV：
      share = sum(r^2 於 (close>vwap)) / sum(r^2 全部)
    要求 close>0。
    """

    name = "rv_share_above_vwap"
    description = "Share of realized variance (log ret) contributed while close>vwap."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close", "vwap"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59").sort_index()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(sample["symbol"].unique())]

        for c in ("close", "vwap"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[(df["close"] > 0)].dropna(subset=["close", "vwap"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df["logc"] = np.log(df["close"])
        df["r"] = df.groupby("symbol", sort=False)["logc"].diff().replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["r"])
        if df.empty:
            out["value"] = pd.NA
            return out

        def per_symbol(g: pd.DataFrame) -> float:
            r2 = g["r"] * g["r"]
            denom = float(r2.sum())
            if denom <= 0:
                return np.nan
            share = float(r2[g["close"] > g["vwap"]].sum()) / denom
            return share

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = RVShareAboveVWAPFeature()
