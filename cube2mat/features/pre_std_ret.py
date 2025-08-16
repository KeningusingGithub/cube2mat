# features/pre_std_ret.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class PreStdRetFeature(BaseFeature):
    """
    盘前 00:00–09:29 内，按 symbol 计算日内简单收益率 ret=close.pct_change() 的标准差；
    若有效收益率个数 < 3，则 NaN。
    输出: ['symbol', 'value']。
    """
    name = "pre_std_ret"
    description = "Pre-market std of intraday returns (pct_change of close) 00:00–09:29; NaN if <3 returns."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns   = ("symbol",)

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
        df = df_full.between_time("00:00", "09:29")
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        syms = set(sample["symbol"].unique())
        df = df[df["symbol"].isin(syms)]
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change()
        df["ret"] = df["ret"].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret"])
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        g = df.groupby("symbol")["ret"]
        stats = g.agg(n="count", std="std")
        val = stats["std"].where(stats["n"] >= 3)

        out = sample[["symbol"]].copy()
        out["value"] = out["symbol"].map(val)
        return out

feature = PreStdRetFeature()
