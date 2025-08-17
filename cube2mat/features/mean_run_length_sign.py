# features/mean_run_length_sign.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class MeanRunLengthSignFeature(BaseFeature):
    """
    Mean run length of non-zero simple-return signs in RTH (combine up & down runs).
    Returns NaN if fewer than two non-zero signs or no runs.
    """

    name = "mean_run_length_sign"
    description = "Average length of consecutive non-zero sign runs (RTH)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _mean_runlen(sign: np.ndarray) -> float:
        s = sign[sign != 0]
        if s.size < 2:
            return np.nan
        runs: list[int] = []
        cur = 1
        for i in range(1, s.size):
            if s[i] == s[i - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        return float(np.mean(runs)) if runs else np.nan

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "close"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59")
        df = df[df.symbol.isin(sample.symbol.unique())].copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            r = (
                g.sort_index()["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            ).to_numpy()
            res[sym] = self._mean_runlen(np.sign(r)) if r.size > 0 else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = MeanRunLengthSignFeature()
