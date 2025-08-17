from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class QnVolLogRetFeature(BaseFeature):
    """
    基于 log 收益的 Qn 稳健尺度估计（Rousseeuw & Croux）：
      Qn = c * Q1( |r_i - r_j|, i<j )，其中 c ≈ 2.2219 使其对正态分布一致。
    有效 r 数 < 3 则 NaN。复杂度 O(n^2)，但 n≈bar 数，通常可接受。
    """
    name = "qn_vol_logret"
    description = "Robust Qn scale (2.2219 * 1st quartile of pairwise |Δr|) for log returns."
    required_full_columns = ("symbol","time","close")
    required_pv_columns   = ("symbol",)
    C = 2.2219

    @staticmethod
    def _qn_scale(arr: np.ndarray) -> float:
        n = arr.size
        if n < 3: return np.nan
        # 两两差的绝对值（上三角）
        diffs = []
        for i in range(n-1):
            di = np.abs(arr[i+1:] - arr[i])
            if di.size:
                diffs.append(di)
        if not diffs:
            return np.nan
        diffs = np.concatenate(diffs)
        q1 = np.quantile(diffs, 0.25)
        if not np.isfinite(q1):
            return np.nan
        return float(QnVolLogRetFeature.C * q1)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full=self.load_full(ctx,date,list(self.required_full_columns))
        sample=self.load_pv(ctx,date,list(self.required_pv_columns))
        if full is None or sample is None: return None

        out=sample[["symbol"]].copy()
        if full.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(full,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df["symbol"].isin(sample["symbol"].unique())]
        if df.empty: out["value"]=pd.NA; return out

        df["close"]=pd.to_numeric(df["close"], errors="coerce")
        df=df[(df["close"]>0)].dropna(subset=["close"]).sort_index()
        if df.empty: out["value"]=pd.NA; return out

        df["log_close"]=np.log(df["close"])
        df["r"]=df.groupby("symbol",sort=False)["log_close"].diff().replace([np.inf,-np.inf],np.nan)
        df=df.dropna(subset=["r"])
        if df.empty: out["value"]=pd.NA; return out

        value = df.groupby("symbol")["r"].apply(lambda s: self._qn_scale(s.values.astype(float)))
        out["value"]=out["symbol"].map(value)
        return out

feature = QnVolLogRetFeature()
