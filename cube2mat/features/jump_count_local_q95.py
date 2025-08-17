# features/jump_count_local_q95.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class JumpCountLocalQ95Feature(BaseFeature):
    """
    09:30–15:59 内，|logret| 超过其 95% 分位数的跳数。
    有效收益 <10 时 NaN。
    """

    name = "jump_count_local_q95"
    description = "Count of |logret| exceeding its 95% session quantile."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)
    q = 0.95

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
        df = df[df.symbol.isin(sample.symbol.unique())]
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            a = (
                np.log(g.sort_index()["close"]).diff().replace([np.inf, -np.inf], np.nan).abs().dropna()
            )
            if len(a) < 10:
                res[sym] = np.nan
                continue
            thr = float(a.quantile(self.q))
            res[sym] = float((a > thr).sum())

        out["value"] = out["symbol"].map(res)
        return out


feature = JumpCountLocalQ95Feature()
