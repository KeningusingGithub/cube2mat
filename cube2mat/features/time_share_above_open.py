# features/time_share_above_open.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class TimeShareAboveOpenFeature(BaseFeature):
    """
    定义锚 = 当日首个有效 open（若缺则首个 close）。
    时间占比 = count(close >= anchor)/count(valid close)。
    """

    name = "time_share_above_open"
    description = "Fraction of bars with close >= session anchor (first open else first close)."
    required_full_columns = ("symbol", "time", "open", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "open", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59").sort_index()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(sample["symbol"].unique())]

        for c in ("open", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        def per_symbol(g: pd.DataFrame) -> float:
            first_open = g["open"].dropna()
            anchor = first_open.iloc[0] if not first_open.empty else g["close"].iloc[0]
            if not np.isfinite(anchor):
                return np.nan
            x = (g["close"] >= anchor).mean()
            return float(x)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = TimeShareAboveOpenFeature()
