# features/partial_corr_absret_volume_time.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class PartialCorrAbsRetVolumeTimeFeature(BaseFeature):
    """
    控制线性时间趋势后，|logret| 与 volume 的偏相关。
    步骤：
      1) r = diff(log(close)); a = |r|
      2) t = 分钟归一化 [0,1]，与 a 对齐
      3) 分别对 a 和 volume 回归 [1,t]，取残差后相关
    样本 <3 或任一方差=0 时 NaN。
    """

    name = "partial_corr_absret_volume_time"
    description = "Partial corr(|logret|, volume | time) within RTH."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

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

        df = self.ensure_et_index(full, "time", ctx.tz).between_time("09:30", "15:59")
        df = df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            r = np.log(g["close"]).diff().replace([np.inf, -np.inf], np.nan)
            a = r.abs().iloc[1:]
            v = g["volume"].iloc[1:]
            t0 = g.index[0]
            tf = ((g.index - t0).total_seconds() / 60.0) / self.TOTAL_MIN
            tf = tf.iloc[1:]
            df2 = pd.DataFrame({"a": a, "v": v, "t": tf}).dropna()
            if len(df2) < 3:
                res[sym] = np.nan
                continue
            n = len(df2)
            X = np.column_stack([np.ones(n), df2["t"].to_numpy()])
            beta_a, *_ = np.linalg.lstsq(X, df2["a"].to_numpy(), rcond=None)
            beta_v, *_ = np.linalg.lstsq(X, df2["v"].to_numpy(), rcond=None)
            res_a = df2["a"].to_numpy() - X @ beta_a
            res_v = df2["v"].to_numpy() - X @ beta_v
            res[sym] = self._corr(res_a, res_v)

        out["value"] = out["symbol"].map(res)
        return out


feature = PartialCorrAbsRetVolumeTimeFeature()
