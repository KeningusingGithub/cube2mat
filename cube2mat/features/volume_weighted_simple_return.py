from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


class VolumeWeightedSimpleReturn(BaseFeature):
    name = "volume_weighted_simple_return"
    description = "Volume-weighted mean of simple returns within RTH: Σ(v_t * (close_t/close_{t-1} - 1))/Σ v_t."
    required_full_columns = ("symbol", "time", "close", "volume")
    required_pv_columns = ("symbol",)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=list(self.required_full_columns))
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            z = _rth(g)[["close", "volume"]].dropna()
            if z.empty or z.shape[0] < 2:
                out[sym] = float("nan")
                continue
            ret = z["close"].astype(float).pct_change().dropna()
            vol = z["volume"].astype(float).reindex(ret.index)
            denom = float(vol.sum())
            num = float((vol * ret).sum())
            out[sym] = num / denom if denom > 0 and np.isfinite(denom) else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = VolumeWeightedSimpleReturn()
