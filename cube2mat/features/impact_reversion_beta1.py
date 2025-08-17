# features/impact_reversion_beta1.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


class ImpactReversionBeta1Feature(BaseFeature):
    """
    OLS slope in: next_ret_simple ~ current volume, within 09:30–15:59.
    Positive beta: higher volume predicts higher next return; negative implies reversion.
    NaN if <8 pairs or Var(volume)=0.
    """
    name = "impact_reversion_beta1"
    description = "OLS slope of next simple return on current volume within 09:30–15:59."
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
        df["close"]  = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if df.empty:
            out["value"] = pd.NA; return out

        res = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.sort_index()
            ret = g["close"].pct_change()
            vol = g["volume"]
            y = ret.shift(-1)  # next simple return
            x = vol
            xy = pd.concat([x, y], axis=1).dropna()
            if len(xy) < 8:
                res[sym] = np.nan
                continue
            x_c = xy.iloc[:,0] - xy.iloc[:,0].mean()
            y_c = xy.iloc[:,1] - xy.iloc[:,1].mean()
            varx = float((x_c * x_c).sum())
            cov  = float((x_c * y_c).sum())
            res[sym] = cov / varx if varx > 0 else np.nan

        out["value"] = out["symbol"].map(res)
        return out


feature = ImpactReversionBeta1Feature()
