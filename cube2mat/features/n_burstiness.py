# features/n_burstiness.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class NBurstinessFeature(BaseFeature):
    """
    09:30–15:59 内，成交笔数的 Burstiness 指数：
      B = (sigma - mu) / (sigma + mu)，mu>0 且 sigma>0 才有效；范围(-1,1)。
    """

    name = "n_burstiness"
    description = "Burstiness of trade counts n across intraday bars: (std-mean)/(std+mean)."
    required_full_columns = ("symbol", "time", "n")
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

        df = df.copy()
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["n"])
        if df.empty:
            out["value"] = pd.NA
            return out

        def per_symbol(g: pd.DataFrame) -> float:
            x = g["n"].astype(float)
            mu = x.mean()
            sd = x.std(ddof=1) if len(x) >= 2 else np.nan
            if not np.isfinite(mu) or not np.isfinite(sd) or mu <= 0 or sd <= 0:
                return np.nan
            return float((sd - mu) / (sd + mu))

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = NBurstinessFeature()
