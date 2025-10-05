from __future__ import annotations

import datetime as dt
from typing import Dict

import pandas as pd

from feature_base import FeatureContext
from trades_feature_base import DATARAW_ROOT, TradesBaseFeature


class TradeSomethingFeature(TradesBaseFeature):
    """示例模板：统计 RTH 内每个 symbol 的指标。"""

    name = "trade_something"
    description = "Example trade-derived metric per symbol"
    default_trade_root = str(DATARAW_ROOT / "us" / "trade")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_trade_columns = ("symbol", "participant_timestamp", "price", "size")

    def process_date(self, ctx: FeatureContext, date: dt.date):
        sample = self.load_pv(ctx, date, columns=["symbol"])
        if sample is None:
            return None
        if sample.empty:
            return pd.DataFrame(columns=["symbol", "value"])

        symbols = list(sample["symbol"].astype(str).unique())

        trade_path = self.get_trade_path(ctx, date)
        if trade_path is None or not trade_path.exists():
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        tz = getattr(ctx, "tz", "America/New_York")
        metric: Dict[str, float] = {}

        for df in self.iterate_trade_batches(
            ctx,
            date,
            columns=list(self.required_trade_columns),
            symbols=symbols,
            batch_size=self.BATCH_SIZE,
        ):
            if df is None or df.empty:
                continue

            df = df.copy()
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["size"] = pd.to_numeric(df["size"], errors="coerce")
            df = df.dropna(subset=["participant_timestamp", "price", "size"])
            if df.empty:
                continue

            df = self.ensure_participant_index(df, tz)
            df = df.between_time(self.RTH_START, self.RTH_END, include_end=False)
            if df.empty:
                continue

            # TODO: 根据具体需求计算指标。这里以成交额为示例。
            df["_dollar"] = df["price"] * df["size"]
            agg = df.groupby("symbol")["_dollar"].sum()

            for sym, value in agg.items():
                metric[sym] = metric.get(sym, 0.0) + float(value)

        out = sample[["symbol"]].copy()
        out["value"] = [metric.get(str(sym), pd.NA) for sym in sample["symbol"]]
        return out


feature = TradeSomethingFeature()

