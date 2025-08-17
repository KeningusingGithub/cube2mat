# features/n_entropy_concentration.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class NEntropyConcentrationFeature(BaseFeature):
    """
    Concentration of trade counts via 1 - normalized Shannon entropy.
    Steps:
      - p_i = n_i / sum(n_i), keep p_i>0
      - H = -sum p_i*log(p_i); Hmax=log(m) where m=len(p_i)
      - value = 1 - H/Hmax in [0,1]; NaN if sum(n)<=0 or m<2
    """
    name = "n_entropy_concentration"
    description = "1 - normalized entropy of n distribution across RTH bars (09:30–15:59)."
    required_full_columns = ("symbol", "time", "n")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, list(self.required_pv_columns))
        if df_full is None or sample is None: return None
        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty: out["value"] = pd.NA; return out

        df = self.ensure_et_index(df_full, "time", ctx.tz).between_time("09:30","15:59")
        if df.empty: out["value"] = pd.NA; return out

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))].copy()
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["n"])
        if df.empty: out["value"] = pd.NA; return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            x = g.sort_index()["n"].astype(float).values
            total = float(np.nansum(x))
            if not np.isfinite(total) or total <= 0:
                res[sym] = np.nan; continue
            p = x / total
            p = p[p > 0]
            m = p.size
            if m < 2:
                res[sym] = np.nan; continue
            H = float(-(p * np.log(p)).sum())
            Hmax = float(np.log(m))
            val = 1.0 - (H / Hmax) if Hmax > 0 else np.nan
            res[sym] = float(np.clip(val, 0.0, 1.0)) if np.isfinite(val) else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = NEntropyConcentrationFeature()
