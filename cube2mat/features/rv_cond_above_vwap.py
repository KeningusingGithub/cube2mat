# features/rv_cond_above_vwap.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RVCondAboveVWAPFeature(BaseFeature):
    """
    仅在 close>vwap 时累积的 RV：∑ r^2 (logret)，对齐到收益终点。
    有效收益 <3 或无满足条件的收益 → NaN。
    """

    name = "rv_cond_above_vwap"
    description = "Sum of r^2 (logret) while close>vwap within RTH."
    required_full_columns = ("symbol", "time", "close", "vwap")
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

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59").copy()
        for c in ("close", "vwap"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "vwap"])
        df = df[df.symbol.isin(sample.symbol.unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            r = np.log(g["close"]).diff().replace([np.inf, -np.inf], np.nan)
            r = r.iloc[1:]
            state = (g["close"] > g["vwap"]).iloc[1:]
            rr = (r * r)[state]
            if rr.dropna().empty:
                res[sym] = np.nan
                continue
            res[sym] = float(rr.sum())

        out["value"] = out["symbol"].map(res)
        return out


feature = RVCondAboveVWAPFeature()
