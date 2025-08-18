# features/shock_calm_time_median_q50.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class ShockCalmTimeMedianQ50Feature(BaseFeature):
    name = "shock_calm_time_median_q50"
    description = (
        "Median bars needed after a |r|≥Q90 event for |r| to drop to ≤ median(|r|) again (log-returns, RTH)."
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
            r = _logret(_rth(g)["close"]).dropna().values
            n = r.size
            if n == 0:
                out[sym] = float("nan")
                continue
            med = float(np.nanmedian(np.abs(r)))
            thr = float(np.nanquantile(np.abs(r), 0.90))
            shocks = np.where(np.abs(r) >= thr)[0]
            times: list[int] = []
            for i in shocks:
                if i + 1 >= n:
                    continue
                h = 1
                while i + h < n and np.abs(r[i + h]) > med:
                    h += 1
                if i + h < n:
                    times.append(h)
            out[sym] = float(np.median(times)) if len(times) > 0 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = ShockCalmTimeMedianQ50Feature()
