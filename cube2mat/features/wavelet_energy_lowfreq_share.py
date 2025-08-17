# features/wavelet_energy_lowfreq_share.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class WaveletEnergyLowfreqShareFeature(BaseFeature):
    """
    Haar 小波分解后，近似系数能量占比：
      - close 对时间做 OLS 去趋势并去均值
      - 进行 L=3 级 Haar 分解，取最终近似 a_L
      - value = sum(a_L^2) / sum(残差^2)
    样本不足或总能量<=0 → NaN。
    """

    name = "wavelet_energy_lowfreq_share"
    description = "Haar-DWT low-frequency energy share of detrended close (L=3)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)
    L = 3

    @staticmethod
    def _haar_levels(y: np.ndarray, L: int) -> np.ndarray | None:
        a = y.copy()
        for _ in range(L):
            n = a.size
            if n < 2:
                return None
            if n % 2 == 1:
                a = np.append(a, a[-1])
                n += 1
            a_next = (a[0::2] + a[1::2]) / np.sqrt(2.0)
            a = a_next
        return a

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
        df = df[df.symbol.isin(sample.symbol.unique())]
        if df.empty:
            out["value"] = pd.NA
            return out
        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        L = int(self.L)
        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            y = g.sort_index()["close"].to_numpy(dtype=float)
            n = y.size
            if n < 8:
                res[sym] = np.nan
                continue
            t = np.linspace(0.0, 1.0, n, endpoint=True)
            X = np.column_stack([np.ones(n), t])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            e = y - X @ beta
            e = e - e.mean()
            tot = float((e * e).sum())
            if tot <= 0 or not np.isfinite(tot):
                res[sym] = np.nan
                continue
            aL = self._haar_levels(e, min(L, int(np.floor(np.log2(max(2, n))))))
            if aL is None or aL.size < 1:
                res[sym] = np.nan
                continue
            low = float((aL * aL).sum())
            res[sym] = float(np.clip(low / tot, 0.0, 1.0))

        out["value"] = out["symbol"].map(res)
        return out


feature = WaveletEnergyLowfreqShareFeature()
