from __future__ import annotations
import datetime as dt, numpy as np, pandas as pd
from feature_base import BaseFeature, FeatureContext


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s.astype(float)).diff()


def _quantile_bins(x: np.ndarray, q: int):
    if x.size == 0:
        return None
    qs = np.linspace(0, 1, q + 1)
    edges = np.unique(np.quantile(x, qs))
    if edges.size < 2:
        return None
    return edges


def _mutual_info(x: np.ndarray, y: np.ndarray, qx: int = 8, qy: int = 8) -> float:
    bx = _quantile_bins(x, qx)
    by = _quantile_bins(y, qy)
    if bx is None or by is None:
        return float("nan")
    ix = np.digitize(x, bx[1:-1], right=False)
    iy = np.digitize(y, by[1:-1], right=False)
    kx, ky = bx.size - 1, by.size - 1
    joint = np.zeros((kx, ky), dtype=float)
    for a, b in zip(ix, iy):
        joint[a, b] += 1.0
    joint_sum = joint.sum()
    if joint_sum <= 0:
        return float("nan")
    px = joint.sum(axis=1) / joint_sum
    py = joint.sum(axis=0) / joint_sum
    pxy = joint / joint_sum
    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log2(pxy[nz] / (px[:, None] * py[None, :])[nz]))
    return float(mi)


class MutualInfoRetVolumeQ8(BaseFeature):
    name = "mutual_info_ret_volume_q8"
    description = "Mutual information I(R;V) in bits between log returns and volume using quantile bins (8×8) within RTH."
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
            out[sym] = _mutual_info(r.values, v, 8, 8)
        res = pv[["symbol"]].copy()
        res["value"] = res["symbol"].map(out)
        return res


feature = MutualInfoRetVolumeQ8()
