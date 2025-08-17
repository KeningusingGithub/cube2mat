# features/vwap_avg_run_length_above.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VWAPAvgRunLengthAboveFeature(BaseFeature):
    """
    Mean run length (in bars) where close > vwap within RTH.
    Returns NaN if no positive runs.
    """

    name = "vwap_avg_run_length_above"
    description = "Average length of contiguous runs with close>vwap in RTH."
    required_full_columns = ("symbol", "time", "close", "vwap")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _mean_run_len(x: np.ndarray) -> float:
        if x.size == 0:
            return np.nan
        runs: list[int] = []
        cnt = 0
        for v in x:
            if v:
                cnt += 1
            elif cnt > 0:
                runs.append(cnt)
                cnt = 0
        if cnt > 0:
            runs.append(cnt)
        return float(np.mean(runs)) if runs else np.nan

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "close", "vwap"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59").copy()
        for c in ("close", "vwap"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "vwap"])
        df = df[df.symbol.isin(sample.symbol.unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            mask = (g.sort_index()["close"] > g["vwap"]).to_numpy()
            res[sym] = self._mean_run_len(mask)

        out["value"] = out["symbol"].map(res)
        return out


feature = VWAPAvgRunLengthAboveFeature()
