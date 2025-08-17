from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class SpectralEntropyCloseFeature(BaseFeature):
    """
    Normalized spectral entropy of de-trended close via FFT:
      - build power spectrum excluding DC; p_i = P_i / sum P
      - H = -sum p_i log(p_i) / log(m) in [0,1]; NaN if m<2 or invalid.
    """
    name = "spectral_entropy_close"
    description = "Normalized spectral entropy (0..1) of detrended close."
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
            y=g.sort_index()["close"].to_numpy(dtype=float)
            n=y.size
            if n<8: res[sym]=np.nan; continue
            t=np.linspace(0.0,1.0,n,endpoint=True)
            X=np.column_stack([np.ones(n), t])
            beta,_=np.linalg.lstsq(X,y,rcond=None)
            e=y - X@beta
            e=e - e.mean()
            powr = (np.fft.rfft(e) * np.fft.rfft(e).conj()).real
            pos = powr[1:]
            m = pos.size
            tot = float(np.sum(pos))
            if m < 2 or tot <= 0 or not np.isfinite(tot):
                res[sym]=np.nan; continue
            p = pos / tot
            p = p[p > 0]
            if p.size < 2:
                res[sym]=np.nan; continue
            H = float(-np.sum(p * np.log(p)) / np.log(m))
            res[sym]=float(np.clip(H, 0.0, 1.0))
        out["value"]=out["symbol"].map(res); return out

feature = SpectralEntropyCloseFeature()
