# features/rv_last15m_share.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RVLast15mShareFeature(BaseFeature):
    """
    尾盘 15 分钟 (15:45–15:59) 对日内 RV 的贡献份额。
    RV=∑ r^2, r=logret。总 RV<=0 或有效收益<3 → NaN。
    """

    name = "rv_last15m_share"
    description = "Share of RV from 15:45–15:59 within RTH."
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

        df = self.ensure_et_index(full, "time", ctx.tz)
        rth = df.between_time("09:30", "15:59")
        rth = rth[rth["symbol"].isin(sample["symbol"].unique())].copy()
        rth["close"] = pd.to_numeric(rth["close"], errors="coerce")
        rth = rth.dropna(subset=["close"])
        if rth.empty:
            out["value"] = pd.NA
            return out

        tail = rth.between_time("15:45", "15:59")
        res: dict[str, float] = {}
        for sym, g in rth.groupby("symbol", sort=False):
            g = g.sort_index()
            r = np.log(g["close"]).diff().replace([np.inf, -np.inf], np.nan).dropna()
            if len(r) < 3:
                res[sym] = np.nan
                continue
            total = float((r * r).sum())
            if total <= 0 or not np.isfinite(total):
                res[sym] = np.nan
                continue
            idx = g.index
            r.index = idx[1:]
            mask = r.index.isin(tail.index)
            share = float((r[mask] ** 2).sum() / total)
            res[sym] = share

        out["value"] = out["symbol"].map(res)
        return out


feature = RVLast15mShareFeature()
