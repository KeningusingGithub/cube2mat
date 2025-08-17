# features/rv_front_loading_score.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class RVFrontLoadingScoreFeature(BaseFeature):
    """Front-loading score for realized variance r^2."""

    name = "rv_front_loading_score"
    description = "2*AUC(cum RV fraction vs time fraction)-1 within RTH."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

    def _start(self, idx: pd.DatetimeIndex) -> pd.Timestamp:
        day = idx[0].date()
        tz = idx.tz
        return pd.Timestamp.combine(day, dt.time(9, 30)).tz_localize(tz)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, list(self.required_full_columns))
        sample = self.load_pv(ctx, date, list(self.required_pv_columns))
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = (
            self.ensure_et_index(full, "time", ctx.tz)
            .between_time("09:30", "15:59")
        )
        df = df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty:
            out["value"] = pd.NA
            return out
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            r = (
                np.log(g["close"])
                .diff()
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(r) < 3:
                res[sym] = np.nan
                continue
            rsq = r * r
            total = float(rsq.sum())
            if not np.isfinite(total) or total <= 0:
                res[sym] = np.nan
                continue
            start = self._start(g.index)
            x = ((g.index - start).total_seconds() / 60.0) / self.TOTAL_MIN
            x = x.iloc[1:]
            y = rsq.cumsum() / total
            x_arr = x.to_numpy()
            y_arr = y.to_numpy()
            if x_arr.size < 2:
                res[sym] = np.nan
                continue
            if x_arr[0] > 0:
                x_arr = np.insert(x_arr, 0, 0.0)
                y_arr = np.insert(y_arr, 0, 0.0)
            if x_arr[-1] < 1.0:
                x_arr = np.append(x_arr, 1.0)
                y_arr = np.append(y_arr, 1.0)
            auc = float(np.trapz(y_arr, x_arr))
            score = 2.0 * auc - 1.0
            res[sym] = float(np.clip(score, -1.0, 1.0))
        out["value"] = out["symbol"].map(res)
        return out


feature = RVFrontLoadingScoreFeature()
