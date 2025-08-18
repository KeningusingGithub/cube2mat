# features/shock_nextbar_reversion_ratio.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class ShockNextBarReversionRatioFeature(BaseFeature):
    name = "shock_nextbar_reversion_ratio"
    description = (
        "Mean of [-sign(r_t)*r_{t+1}/|r_t|] over |r_t|≥Q90 events (log-returns). +值代表平均次棒反转。"
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
            if n < 2:
                out[sym] = float("nan")
                continue
            thr = float(np.nanquantile(np.abs(r), 0.90))
            idx = np.where(np.abs(r) >= thr)[0]
            vals: list[float] = []
            for i in idx:
                if i + 1 >= n:
                    continue
                denom = np.abs(r[i])
                if denom <= 0 or not np.isfinite(denom):
                    continue
                vals.append(-np.sign(r[i]) * r[i + 1] / denom)
            out[sym] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = ShockNextBarReversionRatioFeature()
