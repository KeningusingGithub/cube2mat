from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class TurningPointCountFeature(BaseFeature):
    """
    Count of turning points using sign changes of close differences, with small-move filtering:
      - d = diff(close)
      - threshold = theta * median(|d|), theta=0.1 (if median==0 -> 0)
      - keep |d| > threshold, remove zeros, count sign flips
    """
    name = "turning_point_count"
    description = "Number of local extrema based on filtered sign flips of Δclose (theta=0.1)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)
    theta = 0.1

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(set(sample.symbol.unique()))].copy()
        df["close"]=pd.to_numeric(df["close"],errors="coerce")
        df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        th=float(self.theta); res={}
        for sym,g in df.groupby("symbol",sort=False):
            s=g.sort_index()["close"].to_numpy(dtype=float)
            if s.size<3: res[sym]=np.nan; continue
            d = np.diff(s)
            med = np.median(np.abs(d))
            thr = th * med
            if not np.isfinite(thr): thr = 0.0
            keep = np.abs(d) > thr
            d2 = d[keep]
            if d2.size < 2:
                res[sym] = np.nan; continue
            sign = np.sign(d2)
            sign = sign[sign != 0]
            if sign.size < 2:
                res[sym] = np.nan; continue
            flips = int(np.sum(sign[1:] != sign[:-1]))
            res[sym] = float(flips)
        out["value"]=out["symbol"].map(res); return out

feature = TurningPointCountFeature()
