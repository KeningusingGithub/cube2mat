# features/rq_logret.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RQLogRetFeature(BaseFeature):
    """
    09:30–15:59 内，RQ = (N/3) * sum(r_t^4)，r_t = diff(log(close))，close>0；N=有效 r_t 数。
    用于估计波动的方差（“vol of vol”尺度）。
    """

    name = "rq_logret"
    description = "Realized quarticity (N/3)*sum(r^4) using log returns."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, list(self.required_full_columns))
        sample = self.load_pv(ctx, date, list(self.required_pv_columns))
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        df = df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df[(df["close"] > 0)].dropna(subset=["close"]).sort_index()
        if df.empty:
            out["value"] = pd.NA
            return out

        df["log_close"] = np.log(df["close"])
        df["r"] = df.groupby("symbol", sort=False)["log_close"].diff()
        df["r"] = df["r"].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["r"])
        if df.empty:
            out["value"] = pd.NA
            return out

        def per_symbol(s: pd.Series) -> float:
            n = len(s)
            if n < 2:
                return np.nan
            return float((n / 3.0) * ((s ** 4).sum()))

        value = df.groupby("symbol")["r"].apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = RQLogRetFeature()

