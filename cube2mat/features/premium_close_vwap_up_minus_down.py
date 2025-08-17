# features/premium_close_vwap_up_minus_down.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class PremiumCloseVWAPUpMinusDownFeature(BaseFeature):
    """
    (close - vwap) 在上涨/下跌状态的均值之差：
      diff = mean(close-vwap | ret>0) - mean(close-vwap | ret<0)
    任一侧样本为空则 NaN。
    """

    name = "premium_close_vwap_up_minus_down"
    description = "Mean(close-vwap|up) minus mean(close-vwap|down)."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close", "vwap"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(sample["symbol"].unique())]

        for c in ("close", "vwap"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "vwap"]).sort_index()

        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df["diff"] = df["close"] - df["vwap"]
        df = df.dropna(subset=["ret", "diff"])

        def per_symbol(g: pd.DataFrame) -> float:
            up = g.loc[g["ret"] > 0, "diff"].mean()
            dn = g.loc[g["ret"] < 0, "diff"].mean()
            if not np.isfinite(up) or not np.isfinite(dn):
                return np.nan
            return float(up - dn)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = PremiumCloseVWAPUpMinusDownFeature()
