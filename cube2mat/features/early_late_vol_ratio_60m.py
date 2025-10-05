# features/early_late_vol_ratio_60m.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class EarlyLateVolRatio60mFeature(BaseFeature):
    """
    前 60 分钟 vs 尾盘 60 分钟的收益波动比：
      ratio = std(ret in 09:30–10:29) / std(ret in 15:00–15:59)。
    任一段有效收益数 <2 或分母=0 → NaN。
    """

    name = "early_late_vol_ratio_60m"
    description = "Std(ret)_first60 / Std(ret)_last60 in RTH."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _std_ret(s: pd.Series) -> float:
        if s.size == 0:
            return np.nan
        r = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        return float(r.std(ddof=1)) if len(r) >= 2 else np.nan

    def process_date(self, ctx: FeatureContext, date: dt.date):
        full = self.load_full(ctx, date, list(self.required_full_columns))
        sample = self.load_pv(ctx, date, list(self.required_pv_columns))
        if full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(full, "time", ctx.tz)
        am = df.between_time("09:30", "10:29")
        pm = df.between_time("15:00", "15:59")
        df = df[df["symbol"].isin(sample["symbol"].unique())].copy()
        for part in (am, pm):
            part["close"] = pd.to_numeric(part["close"], errors="coerce")
        am = am.dropna(subset=["close"])
        pm = pm.dropna(subset=["close"])

        res: dict[str, float] = {}
        for sym in sample["symbol"].unique():
            s_am = am[am["symbol"] == sym].sort_index()["close"]
            s_pm = pm[pm["symbol"] == sym].sort_index()["close"]
            sa = self._std_ret(s_am)
            sp = self._std_ret(s_pm)
            if not np.isfinite(sa) or not np.isfinite(sp) or sp == 0:
                res[sym] = np.nan
            else:
                res[sym] = sa / sp

        out["value"] = out["symbol"].map(res)
        return out


feature = EarlyLateVolRatio60mFeature()
