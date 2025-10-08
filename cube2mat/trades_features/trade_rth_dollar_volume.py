from __future__ import annotations

import datetime as dt
import inspect
from typing import Dict

import pandas as pd

from feature_base import FeatureContext
from trades_feature_base import DATARAW_ROOT, TradesBaseFeature


class TradeRthDollarVolume(TradesBaseFeature):
    """计算 RTH 内的 price*size 成交额。"""

    name = "trade_rth_dollar_volume"
    description = "Sum(price*size) per symbol within RTH"
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
        path = self.get_trade_path(ctx, date)
        if path is None or not path.exists():
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        tz = getattr(ctx, "tz", "America/New_York")
        metric: Dict[str, float] = {}

        use_inclusive = "inclusive" in inspect.signature(pd.DataFrame.between_time).parameters

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
            if use_inclusive:
                df = df.between_time(self.RTH_START, self.RTH_END, inclusive="left")
            else:
                df = df.between_time(self.RTH_START, self.RTH_END, include_end=False)
            if df.empty:
                continue

            values = (df["price"] * df["size"]).groupby(df["symbol"]).sum()
            for sym, value in values.items():
                metric[sym] = metric.get(sym, 0.0) + float(value)

        out = sample[["symbol"]].copy()
        out["value"] = [metric.get(str(sym), pd.NA) for sym in sample["symbol"]]
        return out


feature = TradeRthDollarVolume()

