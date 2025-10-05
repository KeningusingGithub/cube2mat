from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

TOT_MIN=389.0

class PartialCorrVWAPDevVolumeTimeFeature(BaseFeature):
    """
    Partial correlation between |(close-vwap)/vwap| and volume, controlling for linear time.
    Steps: build a = |(close-vwap)/vwap| (aligned to bar), t∈[0,1], residualize both on [1,t], corr(resids).
    """
    name = "partial_corr_vwapdev_volume_time"
    description = "Partial corr(|rel VWAP deviation|, volume | time) in RTH."
    required_full_columns = ("symbol","time","close","vwap","volume")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size<3 or y.size<3 or x.size!=y.size: return np.nan
        xc=x-x.mean(); yc=y-y.mean()
        sx=float(np.sqrt(np.sum(xc*xc))); sy=float(np.sqrt(np.sum(yc*yc)))
        if sx<=0 or sy<=0 or not np.isfinite(sx*sy): return np.nan
        return float(np.sum(xc*yc)/(sx*sy))

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close","vwap","volume"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy();
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df = self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("close","vwap","volume"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close","vwap","volume"])
        df=df[df["vwap"]>0]
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            a=((g["close"]-g["vwap"])/g["vwap"]).abs()
            v=g["volume"]
            if len(a)<3: res[sym]=np.nan; continue
            t0=g.index[0]
            tf=pd.Series(((g.index - t0).total_seconds()/60.0)/TOT_MIN, index=g.index)
            D=pd.DataFrame({"a":a,"v":v,"t":tf}).dropna()
            if len(D)<3: res[sym]=np.nan; continue
            n=len(D)
            X=np.column_stack([np.ones(n), D["t"].to_numpy(float)])
            beta_a, *_ = np.linalg.lstsq(X, D["a"].to_numpy(float), rcond=None)
            beta_v, *_ = np.linalg.lstsq(X, D["v"].to_numpy(float), rcond=None)
            ra=D["a"].to_numpy(float) - X@beta_a
            rv=D["v"].to_numpy(float) - X@beta_v
            res[sym]=self._corr(ra,rv)
        out["value"]=out["symbol"].map(res); return out

feature = PartialCorrVWAPDevVolumeTimeFeature()
