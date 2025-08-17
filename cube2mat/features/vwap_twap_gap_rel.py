from __future__ import annotations
import datetime as dt
import numpy as np,pandas as pd
from feature_base import BaseFeature, FeatureContext

class VWAPTWAPGapRelFeature(BaseFeature):
    """
    Session VWAP vs TWAP relative gap:
      vwap_sess = sum(vwap_i * vol_i)/sum(vol_i); twap = mean(close);
      value = (vwap_sess - twap) / twap.
    NaN if twap<=0 or sum(vol)<=0.
    """
    name = "vwap_twap_gap_rel"
    description = "Relative gap between session VWAP and TWAP (mean close)."
    required_full_columns = ("symbol","time","close","vwap","volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        cols=["symbol","time","close","vwap","volume"]
        df=self.load_full(ctx,date,cols); sample=self.load_pv(ctx,date,["symbol"])
        if df is None or sample is None: return None
        out=sample[["symbol"]].copy()
        if df.empty or sample.empty: out["value"]=pd.NA; return out
        df = self.ensure_et_index(df,"time",ctx.tz).between_time("09:30","15:59").copy()
        for c in ("close","vwap","volume"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["close","vwap","volume"])
        df=df[df["volume"]>0]
        df=df[df.symbol.isin(sample.symbol.unique())]
        if df.empty: out["value"]=pd.NA; return out

        res={}
        for sym,g in df.groupby("symbol",sort=False):
            tw=float(g["close"].mean())
            if not np.isfinite(tw) or tw<=0: res[sym]=np.nan; continue
            den=float(g["volume"].sum())
            if den<=0: res[sym]=np.nan; continue
            vwap_sess=float((g["vwap"]*g["volume"]).sum()/den)
            res[sym]=(vwap_sess - tw)/tw
        out["value"]=out["symbol"].map(res); return out

feature = VWAPTWAPGapRelFeature()
