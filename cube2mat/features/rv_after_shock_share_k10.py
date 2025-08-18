# features/rv_after_shock_share_k10.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class RVAfterShockShareK10Feature(BaseFeature):
    name = "rv_after_shock_share_k10"
    description = (
        "Share of daily RV (Σr^2) occurring in the union of 10-bar windows that follow |r|≥Q90 shock bars."
    )
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=["symbol", "time", "close"])
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            r = _logret(_rth(g)["close"]).dropna()
            rv = np.square(r.values)
            n = rv.size
            if n == 0:
                out[sym] = float("nan")
                continue
            thr = float(np.nanquantile(np.abs(r.values), 0.90))
            shocks = np.where(np.abs(r.values) >= thr)[0]
            if shocks.size == 0:
                out[sym] = float("nan")
                continue
            mask = np.zeros(n, dtype=bool)
            for i in shocks:
                a = i + 1
                b = min(i + 10, n - 1)
                if a <= b:
                    mask[a : b + 1] = True
            num = float(rv[mask].sum())
            den = float(rv.sum())
            out[sym] = (num / den) if den > 0 and np.isfinite(den) else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = RVAfterShockShareK10Feature()
