from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if x.size >= 2 else np.nan


class VolumeReturnCoSkewness(BaseFeature):
    name = "volume_return_coskewness"
    description = "Co-skewness E[(V-μV)^2 (R-μR)] / (σV^2 σR) between volume and log returns in RTH."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=["symbol", "time", "close", "volume"])
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            gg = _rth(g)[["close", "volume"]].dropna()
            r = _logret(gg["close"]).dropna()
            if r.empty:
                out[sym] = float("nan")
                continue
            v = gg["volume"].astype(float).reindex(r.index).values
            rv = r.values
            vmu, rmu = float(np.mean(v)), float(np.mean(rv))
            sv, sr = _safe_std(v), _safe_std(rv)
            if not (np.isfinite(sv) and sv > 0 and np.isfinite(sr) and sr > 0):
                out[sym] = float("nan")
                continue
            num = np.mean(((v - vmu) ** 2) * (rv - rmu))
            out[sym] = float(num / (sv ** 2 * sr))
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = VolumeReturnCoSkewness()
