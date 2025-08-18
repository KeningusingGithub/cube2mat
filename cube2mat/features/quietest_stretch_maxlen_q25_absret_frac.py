from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _longest_true_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = cnt = 0
    for v in mask:
        if v:
            cnt += 1
        else:
            best = max(best, cnt)
            cnt = 0
    return max(best, cnt)


class QuietestStretchMaxLenQ25AbsRetFrac(BaseFeature):
    name = "quietest_stretch_maxlen_q25_absret_frac"
    description = (
        "Longest run length (as a fraction of return count) where |r| ≤ Q25(|r|) within RTH (log-returns)."
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
            thr = float(np.nanquantile(np.abs(r), 0.25))
            mask = np.abs(r) <= thr
            L = _longest_true_run(mask)
            out[sym] = float(L / n)
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = QuietestStretchMaxLenQ25AbsRetFrac()
