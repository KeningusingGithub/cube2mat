# features/impact_elasticity_log.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class ImpactElasticityLogFeature(BaseFeature):
    """
    Elasticity of impact: OLS slope in log|ret| ~ log(volume).
    Steps:
      - r = diff(log(close)); use bars with |r|>0 and volume>0
      - y = log(|r|), x = log(volume)
      - beta = Cov(x,y)/Var(x); NaN if <8 points or Var(x)=0.
    """
    name = "impact_elasticity_log"
    description = "OLS slope of log|logret| on log(volume) within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty:
            out["value"] = pd.NA; return out

        df = self.ensure_et_index(df_full, "time", ctx.tz).between_time("09:30","15:59")
        if df.empty:
            out["value"] = pd.NA; return out

        df = df[df["symbol"].isin(set(sample["symbol"].unique()))].copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if df.empty:
            out["value"] = pd.NA; return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            r = np.log(g["close"]).diff().replace([np.inf, -np.inf], np.nan)
            v = g["volume"]
            mask = (r.notna()) & (r.abs() > 0) & (v > 0)
            if mask.sum() < 8:
                res[sym] = np.nan
                continue
            x = np.log(v[mask])
            y = np.log(r[mask].abs())
            x_c = x - x.mean()
            y_c = y - y.mean()
            varx = float((x_c * x_c).sum())
            cov  = float((x_c * y_c).sum())
            res[sym] = cov / varx if varx > 0 else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = ImpactElasticityLogFeature()
