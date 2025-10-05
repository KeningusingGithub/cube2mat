from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteSIPLatencyP95EventOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet），单次流式扫描（直方图近似分位）。
    读取 ['symbol','sip_timestamp','participant_timestamp']，
    在 RTH 内计算“事件级 SIP 延迟”的 p95（单位：毫秒）：
        latency_ms = max(0, (sip_timestamp - participant_timestamp)/1e6)
    负延迟截断为 0；极端值用固定上界 vmax_ms 截断（默认 3000ms，可用 ctx.latency_max_ms 覆盖）。
    输出：['symbol','value'] 与 PV 顺序对齐。
    """
    name = "quote_sip_latency_p95_event_all"
    description = "RTH event p95 of SIP latency (ms), histogram approximation (onefile, single pass)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END = dt.time(16, 0)
    BATCH_SIZE = 500_000

    # 直方图参数
    DEFAULT_MAX_MS = 3000.0
    N_BINS = 1024

    required_pv_columns = ("symbol",)
    required_quote_columns = ("sip_timestamp", "participant_timestamp", "symbol")

    @staticmethod
    def _rth_mask(ts_ns: pd.Series, tz_name: str, start: dt.time, end: dt.time) -> pd.Series:
        ts = pd.to_datetime(ts_ns.astype("Int64"), unit="ns", utc=True)
        et = ts.dt.tz_convert(tz_name)
        h, m = et.dt.hour, et.dt.minute
        ge_start = (h > start.hour) | ((h == start.hour) & (m >= start.minute))
        lt_end = (h < end.hour) | ((h == end.hour) & (m < end.minute))
        return ge_start & lt_end

    @staticmethod
    def _bin_index(x: np.ndarray, vmax: float, n_bins: int) -> np.ndarray:
        x = np.clip(x, 0.0, vmax)
        idx = np.floor((x / vmax) * n_bins).astype(np.int64)
        idx = np.minimum(idx, n_bins - 1)
        return idx

    @staticmethod
    def _pctl_from_hist(counts: np.ndarray, p: float, vmax: float) -> float:
        total = counts.sum()
        if total <= 0:
            return np.nan
        k = int(np.ceil(p * total))
        csum = counts.cumsum()
        b = int(np.searchsorted(csum, k, side="left"))
        return vmax * (b + 1) / len(counts)

    def process_date(self, ctx: FeatureContext, date: dt.date):
        # 1) PV
        sample = self.load_pv(ctx, date, columns=["symbol"])
        if sample is None:
            return None
        if sample.empty:
            return pd.DataFrame(columns=["symbol", "value"])

        # 2) onefile
        root = Path(getattr(ctx, "quote_root", self.default_quote_root))
        day_path = root / f"{date.strftime('%Y%m%d')}.parquet"
        if not day_path.exists():
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        tz_name = getattr(ctx, "tz", "America/New_York")
        vmax = float(getattr(ctx, "latency_max_ms", self.DEFAULT_MAX_MS))

        pf = pq.ParquetFile(str(day_path))
        cols = ["symbol", "sip_timestamp", "participant_timestamp"]

        hist_by: Dict[str, np.ndarray] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            sip = pd.to_numeric(df["sip_timestamp"], errors="coerce")
            par = pd.to_numeric(df["participant_timestamp"], errors="coerce")
            rth = self._rth_mask(par, tz_name, self.RTH_START, self.RTH_END)

            valid = (
                rth
                & sip.replace([np.inf, -np.inf], np.nan).notna()
                & par.replace([np.inf, -np.inf], np.nan).notna()
            )
            if not bool(valid.any()):
                continue

            lat_ms = ((sip - par) / 1e6).astype(float).clip(lower=0.0)
            vals = lat_ms.loc[valid].values
            syms = df.loc[valid, "symbol"].astype(str).values

            # 统计直方图
            bin_idx = self._bin_index(vals, vmax, self.N_BINS)
            sym_codes, uniq_syms = pd.factorize(syms, sort=False)
            combined = sym_codes * self.N_BINS + bin_idx
            u, c = np.unique(combined, return_counts=True)
            u_sym = (u // self.N_BINS).astype(np.int64)
            u_bin = (u % self.N_BINS).astype(np.int64)

            for sidx, bidx, cnt in zip(u_sym, u_bin, c):
                sym = uniq_syms[sidx]
                if sym not in hist_by:
                    hist_by[sym] = np.zeros(self.N_BINS, dtype=np.int64)
                hist_by[sym][bidx] += int(cnt)

        # 由直方图估计 p95
        p95_by: Dict[str, float] = {}
        for sym, counts in hist_by.items():
            val = float(self._pctl_from_hist(counts, 0.95, vmax))
            if np.isfinite(val):
                p95_by[sym] = val

        out = sample[["symbol"]].copy()
        out["value"] = [p95_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteSIPLatencyP95EventOnefileFeature()
