from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class TimeToHalfOpenCloseRetFrac(BaseFeature):
    name = "time_to_half_openclose_ret_frac"
    description = (
        "Fraction of RTH return observations elapsed when cumulative log return "
        "first reaches 50% of |total O→C log return| (direction-aware)."
    )
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
            r = _logret(_rth(g)["close"]).dropna().values
            n = r.size
            if n == 0:
                out[sym] = float("nan")
                continue
            total = float(np.sum(r))
            if not np.isfinite(total) or np.abs(total) < 1e-12:
                out[sym] = float("nan")
                continue
            target = 0.5 * np.abs(total)
            sgn = np.sign(total)
            cr = np.cumsum(r * sgn)
            idx = np.where(cr >= target)[0]
            out[sym] = float((idx[0] + 1) / n) if idx.size > 0 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = TimeToHalfOpenCloseRetFrac()
