# features/xcorr_sign_volume_leadlag_peak.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class XCorrSignVolumeLeadLagPeakFeature(BaseFeature):
    """
    sign(simple ret) 与 volume 的滞后互相关峰值：
      计算 Corr(sign_ret_t, volume_{t+lag})，lag ∈ [-K,K]，返回最大绝对值。
    样本不足时 NaN。
    """

    name = "xcorr_sign_volume_leadlag_peak"
    description = "Peak |corr(sign(ret), volume shift)| over lags [-5,5] in RTH."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)
    K = 5

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 3 or y.size < 3 or x.size != y.size:
            return np.nan
        xc = x - x.mean()
        yc = y - y.mean()
        sx = np.sqrt((xc * xc).sum())
        sy = np.sqrt((yc * yc).sum())
        if sx <= 0 or sy <= 0 or not np.isfinite(sx * sy):
            return np.nan
        return float((xc * yc).sum() / (sx * sy))

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, list(self.required_full_columns))
        sample = self.load_pv(ctx, date, list(self.required_pv_columns))
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59").copy()
        for c in ("close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        df = df[df.symbol.isin(sample.symbol.unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        K = int(self.K)
        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            sgn = np.sign(g["close"].pct_change()).to_numpy(dtype=float)[1:]
            v = g["volume"].to_numpy(dtype=float)[1:]
            n = sgn.size
            if n < K + 3:
                res[sym] = np.nan
                continue
            best = np.nan
            for lag in range(-K, K + 1):
                if lag < 0:
                    x = sgn[-lag:]
                    y = v[: n + lag]
                elif lag > 0:
                    x = sgn[: n - lag]
                    y = v[lag:]
                else:
                    x = sgn
                    y = v
                if x.size < 3 or y.size < 3 or x.size != y.size:
                    continue
                c = self._corr(x, y)
                if np.isfinite(c):
                    if not np.isfinite(best) or abs(c) > abs(best):
                        best = c
            res[sym] = best

        out["value"] = out["symbol"].map(res)
        return out


feature = XCorrSignVolumeLeadLagPeakFeature()
