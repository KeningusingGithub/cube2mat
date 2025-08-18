from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class Next3AbsRetHighVolMinusLowVol(BaseFeature):
    name = "next3_absret_highvol_minus_lowvol"
    description = "E[Σ_{i=1..3}|r_{t+i}| | vol_t≥Q90] − E[Σ_{i=1..3}|r_{t+i}| | vol_t≤Q10], within RTH."
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
            if r.size < 4:
                out[sym] = float("nan")
                continue
            vol = gg["volume"].astype(float).reindex(r.index)
            v90 = float(np.nanquantile(vol.values, 0.90))
            v10 = float(np.nanquantile(vol.values, 0.10))
            hv = np.where(vol.values >= v90)[0]
            lv = np.where(vol.values <= v10)[0]
            def _next3_abs(i: int) -> float:
                j = min(i + 3, r.size - 1)
                return float(np.abs(r.values[i + 1 : j + 1]).sum()) if i + 1 < r.size else np.nan
            hv_vals = [_next3_abs(i) for i in hv if i + 1 < r.size]
            lv_vals = [_next3_abs(i) for i in lv if i + 1 < r.size]
            hvm = np.nanmean(hv_vals) if len(hv_vals) > 0 else np.nan
            lvm = np.nanmean(lv_vals) if len(lv_vals) > 0 else np.nan
            out[sym] = float(hvm - lvm) if np.isfinite(hvm) and np.isfinite(lvm) else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = Next3AbsRetHighVolMinusLowVol()
