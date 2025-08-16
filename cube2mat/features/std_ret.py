# features/std_ret.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class StdRetFeature(BaseFeature):
    """
    09:30–15:59(交易所本地时区，默认 America/New_York) 内，
    先按 symbol 计算日内逐笔/逐bar 简单收益率 ret = close.pct_change()，
    再计算该 symbol 当日 ret 的标准差；若有效收益率个数 < 3，则置为 NaN。
    输出列至少包含: ['symbol', 'value']。
    """
    name = "std_ret"
    description = "Std of intraday simple returns (pct_change of close) between 09:30–15:59; NaN if <3 returns."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
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

        df_intraday = df_intraday.copy()
        df_intraday["close"] = pd.to_numeric(df_intraday["close"], errors="coerce")
        df_intraday = df_intraday.dropna(subset=["close"])
        if df_intraday.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        # 计算日内简单收益率
        df_intraday = df_intraday.sort_index()
        df_intraday["ret"] = (
            df_intraday.groupby("symbol", sort=False)["close"].pct_change()
        )

        # 清理无效/无穷值 —— 不使用 Series.inplace
        df_intraday["ret"] = df_intraday["ret"].replace([np.inf, -np.inf], np.nan)
        df_intraday = df_intraday.dropna(subset=["ret"])
        if df_intraday.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        g = df_intraday.groupby("symbol")["ret"]
        stats = g.agg(n="count", std="std")
        vol = stats["std"].where(stats["n"] >= 3)

        out = sample[["symbol"]].copy()
        out["value"] = out["symbol"].map(vol)
        return out

feature = StdRetFeature()
