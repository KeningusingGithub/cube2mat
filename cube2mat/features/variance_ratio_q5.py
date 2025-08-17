# features/variance_ratio_q5.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VarianceRatioQ5Feature(BaseFeature):
    """
    Variance Ratio with horizon q=5 on intraday log returns within 09:30–15:59.
    VR(q) = Var(Σ_{i=0..q-1} r_{t+i}) / (q * Var(r_t))
    NaN if insufficient (<q+2 returns) or denominator=0.
    """
    name = "variance_ratio_q5"
    description = "Variance Ratio of intraday log returns with q=5 between 09:30–15:59."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)
    q = 5

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()

        if df_full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df_full, time_col="time", tz=ctx.tz).between_time("09:30", "15:59")
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        q = self.q
        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            s = np.log(g.sort_index()["close"]).diff().replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < q + 2:
                res[sym] = np.nan
                continue
            var1 = s.var(ddof=1)
            if not np.isfinite(var1) or var1 <= 0:
                res[sym] = np.nan
                continue
            roll = s.rolling(window=q).sum().dropna()
            if len(roll) < 2:
                res[sym] = np.nan
                continue
            varq = roll.var(ddof=1)
            res[sym] = float(varq / (q * var1)) if (np.isfinite(varq) and varq >= 0) else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = VarianceRatioQ5Feature()
