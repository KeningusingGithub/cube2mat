# features/std_n.py
from __future__ import annotations
import datetime as dt
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class StdNFeature(BaseFeature):
    """
    09:30–15:59(交易所本地时区，默认 America/New_York) 内，
    按 symbol 计算 n(number of trades) 的标准差；若该 symbol 当日有效点 < 3，则置为 NaN。
    输出列至少包含: ['symbol', 'value']。
    """
    name = "std_n"
    description = "Intraday std of trade count (n) between 09:30–15:59; NaN if <3 ticks."
    required_full_columns = ("symbol", "time", "n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        # 读取当日全量与样本
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None

        if df_full.empty or sample.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        # 统一到交易所时区并设为索引
        df_full = self.ensure_et_index(df_full, time_col="time", tz=ctx.tz)

        # 交易时段筛选
        df_intraday = df_full.between_time("09:30", "15:59")
        if df_intraday.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        # 只保留样本内 symbol
        syms = set(sample["symbol"].unique())
        df_intraday = df_intraday[df_intraday["symbol"].isin(syms)]
        if df_intraday.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        # n 转数值并去 NaN
        df_intraday = df_intraday.copy()
        df_intraday["n"] = pd.to_numeric(df_intraday["n"], errors="coerce")
        df_intraday = df_intraday.dropna(subset=["n"])
        if df_intraday.empty:
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        # 兼容所有 pandas：std + count，再把 n<3 的设为 NaN
        g = df_intraday.groupby("symbol")["n"]
        stats = g.agg(n="count", std="std")
        val = stats["std"].where(stats["n"] >= 3)

        # 按样本顺序对齐输出
        out = sample[["symbol"]].copy()
        out["value"] = out["symbol"].map(val)
        return out

# 供 runner 直接加载
feature = StdNFeature()
