# features/volume_up_down_logratio.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VolumeUpDownLogRatioFeature(BaseFeature):
    """
    上/下成交量的对数比： log( (sum vol|ret>0 + eps) / (sum vol|ret<0 + eps) )。
    eps=1e-12 防零除。
    """

    name = "volume_up_down_logratio"
    description = "Log-ratio of volumes on up vs down bars."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)
    EPS = 1e-12

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close", "volume"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59").sort_index()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(sample["symbol"].unique())]

        for c in ("close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret", "volume"])

        def per_symbol(g: pd.DataFrame) -> float:
            up = g.loc[g["ret"] > 0, "volume"].sum()
            dn = g.loc[g["ret"] < 0, "volume"].sum()
            return float(np.log((up + self.EPS) / (dn + self.EPS)))

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = VolumeUpDownLogRatioFeature()
