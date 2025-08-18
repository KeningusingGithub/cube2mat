from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class TransitionEntropySignRet(BaseFeature):
    name = "transition_entropy_signret"
    description = "Conditional entropy H(S_t | S_{t-1}) of sign(log returns) over RTH, normalized by log2(2)=1."
    required_full_columns = ("symbol", "time", "close")
    required_pv_columns = ("symbol",)

    @staticmethod
    def _cond_entropy(s: np.ndarray) -> float:
        s = s[s != 0]
        if s.size < 2:
            return float("nan")
        tr = np.zeros((2, 2), dtype=float)
        prev = (s[:-1] > 0).astype(int)
        nxt = (s[1:] > 0).astype(int)
        for a, b in zip(prev, nxt):
            tr[a, b] += 1.0
        rowsum = tr.sum(axis=1)
        if rowsum.sum() == 0:
            return float("nan")
        H = 0.0
        for i in range(2):
            if rowsum[i] == 0:
                continue
            p_row = tr[i] / rowsum[i]
            p_row = p_row[p_row > 0]
            H += (rowsum[i] / rowsum.sum()) * (-np.sum(p_row * np.log2(p_row)))
        return float(H)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        df = self.load_full(ctx, date, columns=["symbol", "time", "close"])
        pv = self.load_pv(ctx, date, columns=["symbol"])
        if df is None or pv is None:
            return None
        df = self.ensure_et_index(df, "time", ctx.tz)
        out: dict[str, float] = {}
        for sym, g in df.groupby("symbol", observed=True):
            s = np.sign(_logret(_rth(g)["close"]).dropna().values)
            out[sym] = self._cond_entropy(s)
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = TransitionEntropySignRet()
