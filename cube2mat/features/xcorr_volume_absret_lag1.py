from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext

class XCorrVolumeAbsRetLag1Feature(BaseFeature):
    """
    Pearson correlation between volume_t and |logret_{t+1}| within 09:30–15:59.
    NaN if <3 pairs or zero variance.
    """
    name = "xcorr_volume_absret_lag1"
    description = "Corr(volume_t, |logret_{t+1}|) in RTH; lead-lag (volume leads)."
    required_full_columns = ("symbol","time","close","volume")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        if x.size != y.size or x.size < 3:
            return np.nan
        xc = x - x.mean(); yc = y - y.mean()
        sx = float(np.sqrt(np.sum(xc*xc))); sy = float(np.sqrt(np.sum(yc*yc)))
        if sx <= 0 or sy <= 0 or not np.isfinite(sx*sy): return np.nan
        return float(np.sum(xc*yc) / (sx*sy))

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx,date,list(self.required_full_columns))
        sample = self.load_pv(ctx,date,list(self.required_pv_columns))
        if df is None or sample is None: return None
        out = sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out

        df = self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df = df[df.symbol.isin(set(sample.symbol.unique()))].copy()
        for c in ("close","volume"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df = df.dropna(subset=["close","volume"])
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            g=g.sort_index()
            r = np.log(g["close"]).diff().replace([np.inf,-np.inf],np.nan).abs()
            v = g["volume"]
            y = r.shift(-1)  # |ret_{t+1}|
            xy = pd.concat([v, y], axis=1).dropna()
            res[sym] = self._corr(xy.iloc[:,0].to_numpy(), xy.iloc[:,1].to_numpy())
        out["value"]=out["symbol"].map(res); return out

feature = XCorrVolumeAbsRetLag1Feature()
