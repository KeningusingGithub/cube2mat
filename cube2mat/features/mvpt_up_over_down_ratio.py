# features/mvpt_up_over_down_ratio.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class MVPTUpOverDownRatioFeature(BaseFeature):
    """
    上/下“每笔平均量”之比：
      ratio = [sum(vol|ret>0)/sum(n|ret>0)] / [sum(vol|ret<0)/sum(n|ret<0)]
    分母<=0 时 NaN。
    """

    name = "mvpt_up_over_down_ratio"
    description = "Ratio of mean volume per trade on up vs down bars."
    required_full_columns = ("symbol", "time", "close", "volume", "n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, ["symbol", "time", "close", "volume", "n"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(sample["symbol"].unique())]

        for c in ("close", "volume", "n"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "volume", "n"]).sort_index()

        df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["ret"])

        def per_symbol(g: pd.DataFrame) -> float:
            v_up = g.loc[g["ret"] > 0, "volume"].sum()
            n_up = g.loc[g["ret"] > 0, "n"].sum()
            v_dn = g.loc[g["ret"] < 0, "volume"].sum()
            n_dn = g.loc[g["ret"] < 0, "n"].sum()
            if n_up <= 0 or n_dn <= 0:
                return np.nan
            up = v_up / n_up
            dn = v_dn / n_dn
            if dn <= 0:
                return np.nan
            return float(up / dn)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = MVPTUpOverDownRatioFeature()
