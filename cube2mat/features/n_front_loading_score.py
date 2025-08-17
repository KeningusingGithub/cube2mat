# cube2mat/features/n_front_loading_score.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

TOT_MIN = 389.0  # 09:30–15:59

class NFrontLoadingScoreFeature(BaseFeature):
    """
    Front-loading score for trade counts (n):
      - time fraction x in [0,1] from 09:30
      - cumN fraction y = cumsum(n)/sum(n)
      - score = 2 * AUC(y over x) - 1, in [-1,1]; >0 means front-loaded.
      NaN if sum(n)<=0 or <3 points.
    """
    name = "n_front_loading_score"
    description = "2*AUC(cumN vs time fraction)-1 for n in 09:30–15:59; NaN if no trades."
    required_full_columns = ("symbol", "time", "n")
    required_pv_columns = ("symbol",)

    def _start_end(self, idx: pd.DatetimeIndex):
        day = idx[0].date()
        tz = idx.tz
        start = pd.Timestamp.combine(day, dt.time(9,30)).tz_localize(tz)
        end   = pd.Timestamp.combine(day, dt.time(15,59)).tz_localize(tz)
        return start, end

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
            g = g.sort_index()
            if g.empty: res[sym] = np.nan; continue
            start, end = self._start_end(g.index)
            nsum = float(g["n"].sum())
            if not np.isfinite(nsum) or nsum <= 0:
                res[sym] = np.nan; continue
            tf = ((g.index - start).total_seconds() / 60.0) / TOT_MIN
            y  = g["n"].cumsum() / nsum
            # ensure anchors at (0,0) and (1,1)
            x_arr = tf.to_numpy()
            y_arr = y.to_numpy()
            if x_arr.size < 2:
                res[sym] = np.nan; continue
            if x_arr[0] > 0:
                x_arr = np.insert(x_arr, 0, 0.0); y_arr = np.insert(y_arr, 0, 0.0)
            if x_arr[-1] < 1.0:
                x_arr = np.append(x_arr, 1.0); y_arr = np.append(y_arr, 1.0)
            auc = float(np.trapz(y_arr, x_arr))
            score = 2.0 * auc - 1.0
            res[sym] = np.clip(score, -1.0, 1.0)

        out["value"] = out["symbol"].map(res)
        return out

feature = NFrontLoadingScoreFeature()