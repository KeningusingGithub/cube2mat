# features/shock_cluster_density_q90_win5.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class ShockClusterDensityQ90Win5Feature(BaseFeature):
    name = "shock_cluster_density_q90_win5"
    description = (
        "Fraction of |r|≥Q90 shock events that have another shock within ±5 bars (clustering density)."
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
            thr = float(np.nanquantile(np.abs(r), 0.90))
            idx = np.where(np.abs(r) >= thr)[0]
            if idx.size == 0:
                out[sym] = float("nan")
                continue
            flags: list[bool] = []
            for i in idx:
                near = idx[np.abs(idx - i) <= 5]
                flags.append(True if (near.size > 1) else False)
            out[sym] = float(np.mean(flags)) if len(flags) > 0 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = ShockClusterDensityQ90Win5Feature()
