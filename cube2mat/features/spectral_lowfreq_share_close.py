from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class SpectralLowfreqShareCloseFeature(BaseFeature):
    """
    Low-frequency energy share of de-trended close via FFT:
      - y = close (RTH), de-trend by OLS on time + remove mean
      - P = |rfft(y)|^2; exclude DC bin
      - take first p% of positive frequencies (default p=10%), share = sum(low)/sum(all)
    """
    name = "spectral_lowfreq_share_close"
    description = "Share of low-frequency FFT energy of detrended close (p=10%)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)
    p = 0.10

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

        p=float(self.p); res={}
        for sym,g in df.groupby("symbol",sort=False):
            y=g.sort_index()["close"].to_numpy(dtype=float)
            n=y.size
            if n<8: res[sym]=np.nan; continue
            t=np.linspace(0.0,1.0,n,endpoint=True)
            X=np.column_stack([np.ones(n), t])
            beta,_ = np.linalg.lstsq(X,y,rcond=None)
            e=y - X@beta
            e=e - e.mean()
            spec=np.fft.rfft(e)
            powr = (spec*spec.conj()).real
            if powr.size <= 1:
                res[sym]=np.nan; continue
            pos = powr[1:]
            tot = float(np.sum(pos))
            if tot <= 0 or not np.isfinite(tot): res[sym]=np.nan; continue
            m = pos.size
            k = max(1, int(np.floor(p * m)))
            val = float(np.sum(pos[:k]) / tot)
            res[sym]=val
        out["value"]=out["symbol"].map(res); return out

feature = SpectralLowfreqShareCloseFeature()
