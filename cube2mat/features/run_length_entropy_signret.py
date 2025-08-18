from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _run_lengths(s: np.ndarray) -> list[int]:
    s = s[s != 0]
    if s.size == 0:
        return []
    lens: list[int] = []
    curr = s[0]
    L = 1
    for v in s[1:]:
        if v == curr:
            L += 1
        else:
            lens.append(L)
            curr = v
            L = 1
    lens.append(L)
    return lens


class RunLengthEntropySignRet(BaseFeature):
    name = "run_length_entropy_signret"
    description = "Shannon entropy of sign-run length distribution for log returns in RTH, normalized by log2(#unique lengths)."
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
            s = np.sign(_logret(_rth(g)["close"]).dropna().values)
            lens = _run_lengths(s)
            if len(lens) == 0:
                out[sym] = float("nan")
                continue
            vals, cnts = np.unique(lens, return_counts=True)
            p = cnts.astype(float) / cnts.sum()
            H = -np.sum(p * np.log2(p))
            out[sym] = float(H / np.log2(len(vals))) if len(vals) > 1 else 0.0
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = RunLengthEntropySignRet()
