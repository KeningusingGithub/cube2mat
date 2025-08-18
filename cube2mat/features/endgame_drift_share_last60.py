from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class EndgameDriftShareLast60(BaseFeature):
    name = "endgame_drift_share_last60"
    description = "Sum of log returns in 15:00–15:59 divided by total O→C log return (RTH only)."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=list(self.required_full_columns))
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            gr = g.between_time("09:30", "15:59")["close"].dropna()
            r = _logret(gr).dropna()
            if r.empty:
                out[sym] = float("nan")
                continue
            total = float(r.sum())
            if np.abs(total) < 1e-12:
                out[sym] = float("nan")
                continue
            win = g.between_time("15:00", "15:59").index
            num = float(r.loc[r.index.intersection(win)].sum())
            out[sym] = num / total
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = EndgameDriftShareLast60()
