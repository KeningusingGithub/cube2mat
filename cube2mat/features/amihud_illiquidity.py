# features/amihud_illiquidity.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class AmihudIlliquidityFeature(BaseFeature):
    """
    09:30–15:59 内，Amihud 非流动性：
      ILLIQ = mean_t( |ret_t| / (vwap_t * volume_t) )
    其中 ret = close.pct_change()；过滤无效/零或负的 vwap/volume。
    若有效样本<1，则 NaN。
    """
    name = "amihud_illiquidity"
    description = "Amihud illiquidity: mean(|ret| / (vwap*volume)) within 09:30–15:59; NaN if insufficient data."
    required_full_columns = ("symbol", "time", "close", "vwap", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if full is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.copy()
        for c in ("close", "vwap", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["close", "vwap", "volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()
        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change()
        df["ret"] = df["ret"].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret"])

        # 过滤非正的 vwap 或 volume
        df = df[(df["vwap"] > 0) & (df["volume"] > 0)]
        if df.empty:
            out["value"] = pd.NA
            return out

        df["illiq"] = (df["ret"].abs()) / (df["vwap"] * df["volume"])
        value = df.groupby("symbol")["illiq"].mean()

        out["value"] = out["symbol"].map(value)
        return out

feature = AmihudIlliquidityFeature()
