from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class WaveletEnergyHighfreqShareFeature(BaseFeature):
    """
    High-frequency energy share via Haar DWT on detrended close (L=3):
      detrend close by OLS on time, zero-mean; do L-level Haar; low = sum(a_L^2); total=sum(e^2);
      value = 1 - low/total in [0,1]. NaN if insufficient.
    """
    name = "wavelet_energy_highfreq_share"
    description = "Haar-DWT high-frequency energy share of detrended close (L=3)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)
    L=3

    @staticmethod
    def _haar_approx(y: np.ndarray, L: int) -> np.ndarray | None:
        a=y.copy()
        for _ in range(L):
            n=a.size
            if n<2: return None
            if n%2==1:
                a=np.append(a,a[-1]); n+=1
            a=(a[0::2]+a[1::2])/np.sqrt(2.0)
        return a

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df=self.load_full(ctx,date,["symbol","time","close"])
        sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df=self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df=df[df.symbol.isin(sample.symbol.unique())]
        df["close"]=pd.to_numeric(df["close"],errors="coerce"); df=df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out
        L=int(self.L); res={}
        for sym,g in df.groupby("symbol",sort=False):
            y=g.sort_index()["close"].to_numpy(float)
            n=y.size
            if n<8: res[sym]=np.nan; continue
            t=np.linspace(0,1,n)
            X=np.column_stack([np.ones(n), t])
            beta = np.linalg.lstsq(X,y,rcond=None)[0]
            e=y - X@beta; e=e - e.mean()
            tot=float(np.sum(e*e))
            if tot<=0: res[sym]=np.nan; continue
            aL=self._haar_approx(e, L)
            if aL is None or aL.size<1: res[sym]=np.nan; continue
            low=float(np.sum(aL*aL))
            res[sym]=float(np.clip(1.0 - low/tot, 0.0, 1.0))
        out["value"]=out["symbol"].map(res); return out

feature = WaveletEnergyHighfreqShareFeature()
