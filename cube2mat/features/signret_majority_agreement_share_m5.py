# features/signret_majority_agreement_share_m5.py
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


class SignRetMajorityAgreementShareM5Feature(BaseFeature):
    name = "signret_majority_agreement_share_m5"
    description = (
        "Share of bars where sign(r_t) equals the majority sign of the previous 5 nonzero returns (ties/insufficient skipped)."
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
            r = _logret(_rth(g)["close"]).dropna()
            if r.size < 6:
                out[sym] = float("nan")
                continue
            s = np.sign(r.values)
            ok = 0
            hit = 0
            for t in range(5, len(s)):
                prev = s[t - 5 : t]
                prev = prev[prev != 0]
                if prev.size == 0:
                    continue
                maj = np.sign(prev.sum())
                if maj == 0:
                    continue
                ok += 1
                if s[t] == maj:
                    hit += 1
            out[sym] = float(hit / ok) if ok > 0 else float("nan")
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = SignRetMajorityAgreementShareM5Feature()
