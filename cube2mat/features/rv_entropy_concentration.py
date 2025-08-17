from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class RVEntropyConcentrationFeature(BaseFeature):
    """
    1 - normalized Shannon entropy of r^2 distribution in RTH:
      r=diff(log close); p_i=r_i^2 / sum r^2; H=-sum p log p; conc = 1 - H/log(m) in [0,1].
    NaN if <3 returns or RV<=0.
    """
    name = "rv_entropy_concentration"
    description = "1 - normalized entropy of realized-variance contributions."
    required_full_columns = ("symbol","time","close")
    required_pv_columns = ("symbol",)

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

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            r=np.log(g.sort_index()["close"]).diff().replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
            if r.size<3: res[sym]=np.nan; continue
            z=r*r; tot=float(z.sum())
            if tot<=0: res[sym]=np.nan; continue
            p=z/tot; p=p[p>0]
            m=p.size
            if m<2: res[sym]=np.nan; continue
            H=float(-(p*np.log(p)).sum())
            conc=float(1.0 - H/np.log(m))
            res[sym]=float(np.clip(conc,0.0,1.0))
        out["value"]=out["symbol"].map(res); return out

feature = RVEntropyConcentrationFeature()
