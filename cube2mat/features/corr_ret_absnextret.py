# features/corr_ret_absnextret.py
from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class CorrRetAbsNextRetFeature(BaseFeature):
    """
    Pearson corr between simple ret_t and |simple ret_{t+1}| in RTH.
    NaN if <3 pairs or zero variance.
    """
    name = "corr_ret_absnextret"
    description = "Corr(ret_t, |ret_{t+1}|) leverage-effect proxy."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size!=y.size or x.size<3: return np.nan
        xc=x-x.mean(); yc=y-y.mean()
        sx=float(np.sqrt(np.sum(xc*xc))); sy=float(np.sqrt(np.sum(yc*yc)))
        if sx<=0 or sy<=0 or not np.isfinite(sx*sy): return np.nan
        return float(np.sum(xc*yc)/(sx*sy))

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None:return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(sample.symbol.unique())].copy()
        df["close"]=pd.to_numeric(df["close"],errors="coerce")
        df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            r=g.sort_index()["close"].pct_change().replace([np.inf,-np.inf],np.nan)
            x=r
            y=r.shift(-1).abs()
            xy=pd.concat([x,y],axis=1).dropna()
            res[sym]=self._corr(xy.iloc[:,0].to_numpy(), xy.iloc[:,1].to_numpy())
        out["value"]=out["symbol"].map(res); return out

feature = CorrRetAbsNextRetFeature()
