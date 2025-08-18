from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class VolumeReturnCoKurtosis22(BaseFeature):
    name = "volume_return_cokurtosis22"
    description = "Co-kurtosis(2,2): E[(V-μV)^2 (R-μR)^2] / (σV^2 σR^2) for volume and log returns in RTH."
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
            vmu, rmu = np.mean(v), np.mean(rv)
            sv, sr = np.std(v, ddof=1), np.std(rv, ddof=1)
            if sv <= 0 or sr <= 0 or not np.isfinite(sv) or not np.isfinite(sr):
                out[sym] = float("nan")
                continue
            num = np.mean(((v - vmu) ** 2) * ((rv - rmu) ** 2))
            out[sym] = float(num / (sv ** 2 * sr ** 2))
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = VolumeReturnCoKurtosis22()
