# features/avg_trade_size.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class AvgTradeSizeFeature(BaseFeature):
    """
    09:30–15:59 内，按 symbol 计算：
        平均每笔成交量 = sum(volume) / sum(n)
    若当日交易笔数 sum(n) <= 0，则置为 NaN。
    输出列至少包含: ['symbol', 'value']。
    """
    name = "avg_trade_size"
    description = "Average shares per trade between 09:30–15:59: sum(volume)/sum(n); NaN if no trades."
    required_full_columns = ("symbol", "time", "volume", "n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None

        if df_full.empty or sample.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        df_full = self.ensure_et_index(df_full, time_col="time", tz=ctx.tz)
        df_intraday = df_full.between_time("09:30", "15:59")
        if df_intraday.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        syms = set(sample["symbol"].unique())
        df_intraday = df_intraday[df_intraday["symbol"].isin(syms)]
        if df_intraday.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        # 数值化与清洗
        df_intraday = df_intraday.copy()
        df_intraday["volume"] = pd.to_numeric(df_intraday["volume"], errors="coerce")
        df_intraday["n"] = pd.to_numeric(df_intraday["n"], errors="coerce")
        df_intraday = df_intraday.dropna(subset=["volume", "n"])
        if df_intraday.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        agg = df_intraday.groupby("symbol").agg(
            vol=("volume", "sum"),
            trades=("n", "sum"),
        )
        # 保护：trades <= 0 置空
        agg["value"] = np.where(agg["trades"] > 0, agg["vol"] / agg["trades"], np.nan)

        out = sample[["symbol"]].copy()
        out["value"] = out["symbol"].map(agg["value"])
        return out

feature = AvgTradeSizeFeature()