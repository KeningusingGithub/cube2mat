# features/swing_amplitude_median.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _run_amplitudes(sgn: np.ndarray, r: np.ndarray):
    s = pd.Series(sgn).astype(int).replace(0, np.nan).dropna().astype(int)
    if s.empty:
        return []
    starts = (s != s.shift()).cumsum()
    amps: list[float] = []
    for _, loc in s.groupby(starts).groups.items():
        idx = np.array(loc, dtype=int)
        amps.append(float(np.abs(r[idx].sum())))
    return amps


class SwingAmplitudeMedianFeature(BaseFeature):
    name = "swing_amplitude_median"
    description = (
        "Median absolute swing amplitude per sign run (sum of returns over the run, absolute value)."
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
            if r.size == 0:
                out[sym] = float("nan")
                continue
            s = np.sign(r)
            amps = _run_amplitudes(s, r)
            out[sym] = float(np.median(amps)) if len(amps) > 0 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = SwingAmplitudeMedianFeature()
