# features/rvol_ewma_logret_097.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RVolEWMALogret097Feature(BaseFeature):
    """
    Instantaneous rvol via EWMA of r^2 with lambda=0.97 (minute-level).
    Steps:
      r = diff(log(close)); S_t = λ*S_{t-1} + (1-λ)*r_t^2; output sqrt(S_last).
      NaN if <3 returns.
    """
    name = "rvol_ewma_logret_097"
    description = "EWMA instantaneous realized volatility (sqrt of EWMA of r^2) with λ=0.97 within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)
    lam = 0.97

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

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        lam = float(self.lam)
        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            r = np.log(g.sort_index()["close"]).diff().replace([np.inf, -np.inf], np.nan).dropna()
            if len(r) < 3:
                res[sym] = np.nan
                continue
            s_last = None
            for val in (r * r).values:
                if not np.isfinite(val):
                    continue
                if s_last is None:
                    s_last = val
                else:
                    s_last = lam * s_last + (1 - lam) * val
            res[sym] = float(np.sqrt(s_last)) if s_last is not None else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = RVolEWMALogret097Feature()
