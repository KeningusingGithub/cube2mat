# features/jump_count_local_k3_win21.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class JumpCountLocalK3Win21Feature(BaseFeature):
    """
    基于局部 MAD 的跳数：
      r = diff(log(close))；sigma_local = 1.4826*MAD(|r|) 右对齐滚动窗21。
      计数 |r_t| > k*sigma_local, k=3。有效收益 <21 或无有效 sigma → NaN。
    """

    name = "jump_count_local_k3_win21"
    description = "Count of |logret| > 3×local MAD scale (win=21)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)
    k = 3.0
    window = 21

    @staticmethod
    def _mad(x: pd.Series) -> float:
        m = x.median()
        return 1.4826 * (x - m).abs().median()

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
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            out["value"] = pd.NA
            return out

        k = float(self.k)
        win = int(self.window)
        res: dict[str, float] = {}
        for sym, g in df.groupby("symbol", sort=False):
            r = np.log(g.sort_index()["close"]).diff().replace([np.inf, -np.inf], np.nan)
            a = r.abs()
            if a.dropna().shape[0] < win:
                res[sym] = np.nan
                continue
            sigma = a.rolling(win).apply(lambda s: self._mad(pd.Series(s).dropna()), raw=False)
            mask = (sigma > 0) & a.notna()
            if mask.sum() < 1:
                res[sym] = np.nan
                continue
            cnt = int(((a > k * sigma) & mask).sum())
            res[sym] = float(cnt)

        out["value"] = out["symbol"].map(res)
        return out


feature = JumpCountLocalK3Win21Feature()
