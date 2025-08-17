# features/reversal_prob_after_big_down.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class ReversalProbAfterBigDownFeature(BaseFeature):
    """
    大幅下跌后的反转概率：
      取当日 simple return 的 10% 分位数 q10，触发集 B={t: ret_t <= q10}；
      统计 P(ret_{t+1}>0 | t∈B)。触发样本<3 则 NaN。
    """

    name = "reversal_prob_after_big_down"
    description = "P(next ret > 0 | current ret in bottom 10%)."
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

        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df["ret_next"] = df.groupby("symbol", sort=False)["ret"].shift(-1)
        df = df.dropna(subset=["ret", "ret_next"])

        def per_symbol(g: pd.DataFrame) -> float:
            q10 = np.quantile(g["ret"].values, 0.1)
            B = g["ret"] <= q10
            k = B.sum()
            if k < 3:
                return np.nan
            p = (g.loc[B, "ret_next"] > 0).mean()
            return float(p)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = ReversalProbAfterBigDownFeature()
