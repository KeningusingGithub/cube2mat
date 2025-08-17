from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


class PremktToRTHRetCorrFeature(BaseFeature):
    """
    Approximate linkage between pre-market and RTH returns (single-day proxy):
      value = sign(sum_pre_logret) * sign(sum_rth_logret)
    in {-1, 0, +1}; NaN if either segment lacks >=1 return.
      Pre: 00:00–09:29; RTH: 09:30–15:59.
    """
    name = "premkt_to_rth_ret_corr"
    description = "Sign consistency proxy between pre-market and RTH total log returns."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _seg(df: pd.DataFrame, start: str, end: str) -> pd.Series:
        seg = df.between_time(start, end)["close"]
        r = np.log(seg).diff().replace([np.inf,-np.inf],np.nan).dropna()
        return r

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()

        if df.empty or sample.empty:
            out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz)
        df=df[df["symbol"].isin(sample["symbol"].unique())].copy()
        df["close"]=pd.to_numeric(df["close"],errors="coerce"); df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            pre = self._seg(g, "00:00", "09:29")
            rth = self._seg(g, "09:30", "15:59")
            if len(pre)<1 or len(rth)<1:
                res[sym]=np.nan; continue
            s = np.sign(pre.sum()) * np.sign(rth.sum())
            res[sym] = float(s)
        out["value"]=out["symbol"].map(res); return out


feature = PremktToRTHRetCorrFeature()
