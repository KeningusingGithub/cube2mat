# features/volume_entropy_concentration.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VolumeEntropyConcentrationFeature(BaseFeature):
    """
    09:30–15:59 内，成交量的“1-归一化熵”：
      p_i = vol_i / sum vol；H = -sum p_i log p_i；H_norm = H / log(N)；value = 1 - H_norm。
    越接近 1 表示越集中；若 sum(vol)<=0 或 N<2，则 NaN。
    """

    name = "volume_entropy_concentration"
    description = (
        "1 - normalized Shannon entropy of volume distribution across intraday bars."
    )
    required_full_columns = ("symbol", "time", "volume")
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
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        def per_symbol(g: pd.DataFrame) -> float:
            N = len(g)
            tot = g["volume"].sum()
            if N < 2 or not np.isfinite(tot) or tot <= 0:
                return np.nan
            p = (g["volume"] / tot).values
            # 过滤 p_i=0 的项（0*log0 视为 0）
            p = p[p > 0]
            if len(p) == 0:
                return np.nan
            H = -(p * np.log(p)).sum()
            H_norm = H / np.log(N)
            if not np.isfinite(H_norm) or H_norm < 0:
                return np.nan
            return float(1.0 - min(1.0, H_norm))

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = VolumeEntropyConcentrationFeature()
