# features/reversal_prob_after_big_up.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class ReversalProbAfterBigUpFeature(BaseFeature):
    """
    大幅上涨后的反转概率：
      取当日 simple return 的 90% 分位数 q90，触发集 A={t: ret_t >= q90}；
      统计 P(ret_{t+1}<0 | t∈A)。触发样本<3 则 NaN。
    """

    name = "reversal_prob_after_big_up"
    description = "P(next ret < 0 | current ret in top 10% by magnitude and positive)."
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
            q90 = np.quantile(g["ret"].values, 0.9)
            A = g["ret"] >= q90
            k = A.sum()
            if k < 3:
                return np.nan
            p = (g.loc[A, "ret_next"] < 0).mean()
            return float(p)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = ReversalProbAfterBigUpFeature()
