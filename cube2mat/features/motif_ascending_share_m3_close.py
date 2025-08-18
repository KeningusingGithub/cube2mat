from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


class MotifAscendingShareM3Close(BaseFeature):
    name = "motif_ascending_share_m3_close"
    description = "Share of 3-bar windows where close is strictly increasing (ordinal motif [0,1,2]) within RTH."
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
            x = _rth(g)["close"].astype(float).dropna().values
            n = x.size
            if n < 3:
                out[sym] = float("nan")
                continue
            cnt = 0
            for i in range(n - 2):
                if x[i] < x[i + 1] < x[i + 2]:
                    cnt += 1
            out[sym] = float(cnt / (n - 2))
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = MotifAscendingShareM3Close()
