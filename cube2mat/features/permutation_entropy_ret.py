# features/permutation_entropy_ret.py
from __future__ import annotations
import datetime as dt
import math
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class PermutationEntropyRetFeature(BaseFeature):
    """
    09:30–15:59 内 logret 的排列熵 (m=4, tau=1)，归一化至 [0,1]。
    有效收益长度 < m+1 时 NaN。
    """

    name = "permutation_entropy_ret"
    description = "Normalized permutation entropy of log returns (m=4, tau=1)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)
    m = 4
    tau = 1

    def _perm_entropy(self, x: np.ndarray, m: int, tau: int) -> float:
        n = x.size
        L = n - (m - 1) * tau
        if L <= 1:
            return np.nan
        counts: dict[tuple[int, ...], int] = {}
        for i in range(L):
            seg = x[i : i + (m * tau) : tau]
            ranks = np.argsort(np.argsort(seg, kind="mergesort"), kind="mergesort")
            key = tuple(ranks.tolist())
            counts[key] = counts.get(key, 0) + 1
        cnt = np.array(list(counts.values()), dtype=float)
        p = cnt / cnt.sum()
        p = p[p > 0]
        if p.size < 2:
            return np.nan
        H = float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))
        return float(np.clip(H, 0.0, 1.0))

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
        df = df[df.symbol.isin(sample.symbol.unique())].copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        m = int(self.m)
        tau = int(self.tau)
        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            r = (
                np.log(g.sort_index()["close"]).diff().replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            )
            res[sym] = self._perm_entropy(r, m, tau)

        out["value"] = out["symbol"].map(res)
        return out


feature = PermutationEntropyRetFeature()
