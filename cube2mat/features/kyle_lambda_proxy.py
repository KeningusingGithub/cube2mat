# features/kyle_lambda_proxy.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class KyleLambdaProxyFeature(BaseFeature):
    """
    Kyle's lambda proxy via OLS slope of |logret| on dollar volume (vwap*volume).
    Steps:
      - r = diff(log(close)) in 09:30–15:59, y = |r|
      - x = vwap * volume (to approximate dollar volume)
      - beta = Cov(x,y) / Var(x); NaN if <8 points or Var(x)=0.
    """
    name = "kyle_lambda_proxy"
    description = "OLS slope of |log returns| on dollar volume (vwap*volume) within 09:30–15:59."
    required_full_columns = ("symbol", "time", "close", "vwap", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df_full = self.load_full(ctx, date, columns=list(self.required_full_columns))
        sample  = self.load_pv(ctx, date, columns=list(self.required_pv_columns))
        if df_full is None or sample is None:
            return None
        out = sample[["symbol"]].copy()
        if df_full.empty or sample.empty:
            out["value"] = pd.NA
            return out

        df = self.ensure_et_index(df_full, "time", ctx.tz).between_time("09:30", "15:59")
        if df.empty:
            out["value"] = pd.NA
            return out
        df = df[df["symbol"].isin(set(sample["symbol"].unique()))]
        if df.empty:
            out["value"] = pd.NA
            return out

        for col in ("close", "vwap", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close", "vwap", "volume"])
        if df.empty:
            out["value"] = pd.NA
            return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            r = np.log(g["close"]).diff().replace([np.inf, -np.inf], np.nan)
            y = r.abs()
            x = g["vwap"] * g["volume"]
            df_xy = pd.concat([x, y], axis=1).dropna()
            if len(df_xy) < 8:
                res[sym] = np.nan
                continue
            x_c = df_xy.iloc[:,0] - df_xy.iloc[:,0].mean()
            y_c = df_xy.iloc[:,1] - df_xy.iloc[:,1].mean()
            varx = float((x_c * x_c).sum())
            cov  = float((x_c * y_c).sum())
            res[sym] = cov / varx if varx > 0 else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = KyleLambdaProxyFeature()
