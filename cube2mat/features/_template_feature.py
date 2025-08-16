# features/_template_feature.py
from __future__ import annotations
import datetime as dt
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class MyNewFeature(BaseFeature):
    name = "my_new_feature"  # 输出目录名：cube2mat/my_new_feature
    description = "示例：把前一日收盘复制成 value（只是演示结构，不具备业务意义）"
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        # 读取数据
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None

        # 建议：尽量先过滤时间/列，最后 groupby，避免 apply
        df_full = self.ensure_et_index(df_full, time_col="time", tz=ctx.tz)
        # 示例：取 09:30 之前的最后一个 close（纯演示）
        pre_open = df_full.between_time("09:00", "09:29").dropna(subset=["close"])
        last_close = pre_open.groupby("symbol", observed=True)["close"].last()

        out = sample[["symbol"]].copy()
        out["value"] = out["symbol"].map(last_close)
        return out

# 让 runner 直接拿到实例
feature = MyNewFeature()
