# features/max_drawdown_depth_per_bar.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


class MaxDrawdownDepthPerBarFeature(BaseFeature):
    name = "max_drawdown_depth_per_bar"
    description = (
        "Maximum drawdown depth divided by its duration in bars (uses close; depth as positive number)."
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
            s = _rth(g)["close"].astype(float).dropna().values
            n = s.size
            if n < 2:
                out[sym] = float("nan")
                continue
            run_max = np.maximum.accumulate(s)
            dd = s / run_max - 1.0
            trough = int(np.argmin(dd))
            if trough == 0:
                out[sym] = float("nan")
                continue
            peak = int(np.argmax(s[: trough + 1]))
            depth = -float(dd[trough])
            dur = trough - peak
            out[sym] = float(depth / dur) if dur > 0 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = MaxDrawdownDepthPerBarFeature()
