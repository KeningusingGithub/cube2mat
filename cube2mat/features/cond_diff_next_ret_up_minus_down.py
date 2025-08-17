# features/cond_diff_next_ret_up_minus_down.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class CondDiffNextRetUpMinusDownFeature(BaseFeature):
    """
    条件差：E[ret_{t+1}|ret_t>0] - E[ret_{t+1}|ret_t<0]。
    两侧触发样本均需≥3，否则 NaN。
    """

    name = "cond_diff_next_ret_up_minus_down"
    description = "Difference between next-return means after up vs down bars."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59").sort_index()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(sample["symbol"].unique())]

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df["ret_next"] = df.groupby("symbol", sort=False)["ret"].shift(-1)
        df = df.dropna(subset=["ret", "ret_next"])

        def per_symbol(g: pd.DataFrame) -> float:
            up = g.loc[g["ret"] > 0, "ret_next"]
            dn = g.loc[g["ret"] < 0, "ret_next"]
            if up.size < 3 or dn.size < 3:
                return np.nan
            return float(up.mean() - dn.mean())

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = CondDiffNextRetUpMinusDownFeature()
