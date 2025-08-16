# features/pre_mean_close.py
from __future__ import annotations
import datetime as dt
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class PreMeanCloseFeature(BaseFeature):
    """
    盘前 00:00–09:29 内，按 symbol 计算 close 的均值；若该 symbol 当日无有效点，则 NaN。
    输出: ['symbol', 'value']。
    """
    name = "pre_mean_close"
    description = "Pre-market mean(close) between 00:00–09:29."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns   = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        # 读取数据
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None

        if df_full.empty or sample.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        # 时区与时间筛选
        df_full = self.ensure_et_index(df_full, time_col="time", tz=ctx.tz)
        df = df_full.between_time("00:00", "09:29")
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        # 样本内 symbol
        syms = set(sample["symbol"].unique())
        df = df[df["symbol"].isin(syms)]
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        # 数值化
        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        meanv = df.groupby("symbol")["close"].mean()
        out = sample[["symbol"]].copy()
        out["value"] = out["symbol"].map(meanv)
        return out

# 供 runner 直接加载
feature = PreMeanCloseFeature()
