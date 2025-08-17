# features/range_up_down_ratio.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RangeUpDownRatioFeature(BaseFeature):
    """
    以 (high-low) 作为区间尺度，对应上涨/下跌状态的总和之比：
      ratio = sum(range|ret>0) / sum(range|ret<0)
    任一侧分母<=0 则 NaN。
    """

    name = "range_up_down_ratio"
    description = "Sum(high-low) on up bars divided by that on down bars."
    required_full_columns = ("symbol", "time", "high", "low", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "high", "low", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(sample["symbol"].unique())]

        for c in ("high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["high", "low", "close"]).sort_index()
        df["range"] = df["high"] - df["low"]

        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret", "range"])

        def per_symbol(g: pd.DataFrame) -> float:
            up = g.loc[g["ret"] > 0, "range"].sum()
            dn = g.loc[g["ret"] < 0, "range"].sum()
            if dn <= 0:
                return np.nan
            return float(up / dn)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = RangeUpDownRatioFeature()
