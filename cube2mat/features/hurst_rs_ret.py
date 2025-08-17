from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

def _hurst_rs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    # candidate block sizes (minute级别会话 ~389；选择几何序)
    sizes = [5, 10, 20, 40, 80, 160]
    sizes = [m for m in sizes if m <= n // 2 and m >= 5]
    RS_pts = []
    for m in sizes:
        k = n // m
        if k < 1:
            continue
        rs_vals = []
        for i in range(k):
            seg = x[i*m:(i+1)*m]
            seg = seg - seg.mean()
            Z = np.cumsum(seg)
            R = float(Z.max() - Z.min())
            S = float(seg.std(ddof=1))
            if S > 0 and np.isfinite(R) and np.isfinite(S):
                rs_vals.append(R / S)
        if len(rs_vals) > 0:
            RS_pts.append((m, float(np.mean(rs_vals))))
    if len(RS_pts) < 2:
        return np.nan
    logs = np.log([p[0] for p in RS_pts])
    logr = np.log([p[1] for p in RS_pts])
    try:
        H = float(np.polyfit(logs, logr, 1)[0])
    except Exception:
        H = np.nan
    return H

class HurstRSRetFeature(BaseFeature):
    """
    Hurst exponent via R/S analysis on intraday log returns in RTH.
    Uses block sizes {5,10,20,40,80,160} as available; OLS slope of log(R/S)~log(m).
    """
    name = "hurst_rs_ret"
    description = "R/S Hurst exponent estimated from intraday log returns."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(set(sample.symbol.unique()))]
        if df.empty: out["value"]=pd.NA; return out
        df["close"]=pd.to_numeric(df["close"],errors="coerce"); df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            r = np.log(g.sort_index()["close"]).diff().replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
            if r.size < 20:
                res[sym]=np.nan; continue
            res[sym] = _hurst_rs(r)
        out["value"]=out["symbol"].map(res); return out

feature = HurstRSRetFeature()
