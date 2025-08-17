# features/volume_front_loading_score.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class VolumeFrontLoadingScoreFeature(BaseFeature):
    """
    前置/后置成交量偏移得分（-1~+1）：
      以时间归一化 tau∈[0,1] 与累积量份额 c(tau) 构成曲线，计算 AUC = ∫ c d tau（梯形法），
      score = 2*AUC - 1。>0 前置（早段更快累积），<0 后置。
    若 sum(vol)<=0 或样本<2，则 NaN。
    """

    name = "volume_front_loading_score"
    description = "2*AUC(cumVolFraction vs timeFraction) - 1; positive=front-loaded."
    required_full_columns = ("symbol", "time", "volume")
    required_pv_columns = ("symbol",)

    TOTAL_MIN = (
        pd.Timedelta("15:59:00") - pd.Timedelta("09:30:00")
    ).total_seconds() / 60.0

    @staticmethod
    def _tau_minutes(idx: pd.DatetimeIndex) -> np.ndarray:
        tmin = (
            idx - idx.normalize() - pd.Timedelta("09:30:00")
        ).total_seconds() / 60.0
        return tmin

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
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        df = df.sort_index()

        def per_symbol(g: pd.DataFrame) -> float:
            v = g["volume"].astype(float).values
            tot = v.sum()
            if not np.isfinite(tot) or tot <= 0 or len(v) < 2:
                return np.nan
            cfrac = np.cumsum(v) / tot
            tau = self._tau_minutes(g.index) / self.TOTAL_MIN
            # 拼接起点(0,0)与终点(1,1)进行梯形积分，避免起末缺失偏差
            tau_ext = np.concatenate(([0.0], tau, [1.0]))
            c_ext = np.concatenate(([0.0], cfrac, [1.0]))
            # 去重并排序（防止时间重复）
            order = np.argsort(tau_ext)
            tau_ext = tau_ext[order]
            c_ext = c_ext[order]
            # 梯形积分
            dtau = np.diff(tau_ext)
            AUC = np.sum(0.5 * (c_ext[1:] + c_ext[:-1]) * dtau)
            score = 2.0 * AUC - 1.0
            return float(score)

        value = df.groupby("symbol").apply(per_symbol)
        out["value"] = out["symbol"].map(value)
        return out


feature = VolumeFrontLoadingScoreFeature()
