# features/n_front_loading_score.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class NFrontLoadingScoreFeature(BaseFeature):
    """
    Front-loading score for trade counts:
    y = cumsum(n) / sum(n) vs x = time fraction in [0, 1];
    score = 2 * AUC(y over x) - 1 in [-1, 1].
    """

    name = "n_front_loading_score"
    description = "2*AUC(cum n fraction vs time fraction)-1 in RTH."
    required_full_columns = ("symbol", "time", "n")
    required_pv_columns = ("symbol",)

    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, ["symbol", "time", "n"])
        sample = self.load_pv(ctx, date, ["symbol"])
        if df is None or sample is None:
            return None

        out = sample[["symbol"]].copy()
        if df.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df, "time", ctx.tz).between_time("09:30", "15:59").copy()
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        df = df.dropna(subset=["n"])
        df = df[df.symbol.isin(sample.symbol.unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            total = float(g["n"].sum())
            if total <= 0:
                res[sym] = np.nan
                continue
            start = g.index[0]
            x = ((g.index - start).total_seconds() / 60.0) / self.TOTAL_MIN
            y = g["n"].cumsum() / total
            xa = x.to_numpy()
            ya = y.to_numpy()
            if xa[0] > 0:
                xa = np.insert(xa, 0, 0.0)
                ya = np.insert(ya, 0, 0.0)
            if xa[-1] < 1:
                xa = np.append(xa, 1.0)
                ya = np.append(ya, 1.0)
            auc = float(np.trapz(ya, xa))
            score = 2 * auc - 1
            res[sym] = float(np.clip(score, -1.0, 1.0))

        out["value"] = out["symbol"].map(res)
        return out


feature = NFrontLoadingScoreFeature()
