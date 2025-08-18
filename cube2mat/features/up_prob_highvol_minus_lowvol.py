from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class UpProbHighVolMinusLowVol(BaseFeature):
    name = "up_prob_highvol_minus_lowvol"
    description = "ΔP(up): P(r>0 | volume≥Q90) − P(r>0 | volume≤Q10) using RTH log returns aligned to volume at time t."
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
            vol = gg["volume"].astype(float).reindex(r.index)
            v90 = float(np.nanquantile(vol.values, 0.90))
            v10 = float(np.nanquantile(vol.values, 0.10))
            hv = vol.values >= v90
            lv = vol.values <= v10
            ph = (r.values[hv] > 0).mean() if hv.sum() > 0 else np.nan
            pl = (r.values[lv] > 0).mean() if lv.sum() > 0 else np.nan
            out[sym] = float(ph - pl) if np.isfinite(ph) and np.isfinite(pl) else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = UpProbHighVolMinusLowVol()
