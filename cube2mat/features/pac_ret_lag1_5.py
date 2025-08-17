from __future__ import annotations
import datetime as dt
import numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _acf(x: np.ndarray, m: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < m + 2:
        return np.full(m+1, np.nan)
    xc = x - x.mean()
    den = float(np.sum(xc*xc))
    if den <= 0 or not np.isfinite(den):
        return np.full(m+1, np.nan)
    r = np.empty(m+1, dtype=float)
    for k in range(m+1):
        r[k] = float(np.sum(xc[k:] * xc[:n-k])) / den
    return r


def _pacf_durbin_levinson(r: np.ndarray, m: int) -> np.ndarray:
    # r: acf with r[0]=1; returns pacf[1..m]
    pacf = np.zeros(m+1, dtype=float)
    phi_prev = np.zeros(m+1, dtype=float)
    v = 1.0
    for k in range(1, m+1):
        if k == 1:
            phi = r[1]
        else:
            num = r[k] - np.sum(phi_prev[1:k] * r[1:k][::-1])
            if v <= 0 or not np.isfinite(v):
                phi = np.nan
            else:
                phi = num / v
        pacf[k] = phi
        # update phi_prev
        if k > 1 and np.isfinite(phi):
            phi_new = phi_prev.copy()
            for j in range(1, k):
                phi_new[j] = phi_prev[j] - phi * phi_prev[k-j]
            phi_new[k] = phi
            phi_prev = phi_new
        else:
            phi_prev[k] = phi
        # update v
        if np.isfinite(phi):
            v = v * (1.0 - phi*phi)
        else:
            v = np.nan
    return pacf


class PACRetLag1to5Feature(BaseFeature):
    """
    Partial autocorrelation of log returns at lags 1..5 via Durbin-Levinson.
    Output: index of the first significant lag in {1..5} (two-sided 95% ~ 1.96/sqrt(n)); 0 if none.
    """
    name = "pac_ret_lag1_5"
    description = "First significant PACF lag (1..5) for log returns; 0 if none. Significance ~ 1.96/sqrt(n)."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)
    maxlag = 5

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx,date,list(self.required_full_columns))
        sample = self.load_pv(ctx,date,list(self.required_pv_columns))
        if df is None or sample is None: return None
        out = sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df = self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59")
        df = df[df.symbol.isin(set(sample.symbol.unique()))]
        if df.empty: out["value"]=pd.NA; return out
        df = df.copy(); df["close"]=pd.to_numeric(df["close"],errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty: out["value"]=pd.NA; return out

        m = int(self.maxlag); res={}
        for sym,g in df.groupby("symbol",sort=False):
            r = np.log(g.sort_index()["close"]).diff().replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
            n = r.size
            if n < m + 2: res[sym]=np.nan; continue
            rho = _acf(r, m)  # rho[0..m]
            if not np.isfinite(rho).all(): rho = np.nan_to_num(rho, nan=0.0)
            rho[0] = 1.0
            pacf = _pacf_durbin_levinson(rho, m)  # pacf[0..m], use 1..m
            thr = 1.96 / np.sqrt(n)
            k0 = 0
            for k in range(1, m+1):
                if np.isfinite(pacf[k]) and abs(pacf[k]) > thr:
                    k0 = k; break
            res[sym] = float(k0)
        out["value"] = out["symbol"].map(res); return out

feature = PACRetLag1to5Feature()
