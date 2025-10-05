# quote_features/imbalance_spread_coupling.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


class QuoteImbSpreadCouplingOnefileFeature(QuoteBaseFeature):
    """
    Onefile 专用（每天一个 {YYYYMMDD}.parquet），两次流式扫描：
    Pass1：用每 symbol 的直方图近似估计 p90(rel_spread)，
    Pass2：计算事件均值 coupling = imb * (1 - rel / p90)，其中 rel = 2*(ask-bid)/(ask+bid) 截断为[0, +∞)。
    仅统计 RTH(09:30–16:00 ET) 内有效记录；对 size_sum>0、denom>0 的样本生效。
    输出：['symbol','value'] 与 PV 顺序对齐。
    """
    name = "quote_imbalance_spread_coupling_all"
    description = "RTH mean of SizeImbalance × (1 - rel_spread / p90(rel_spread)) per symbol (onefile, two-pass)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END   = dt.time(16,  0)
    BATCH_SIZE = 500_000

    REL_MAX = 0.50
    N_BINS  = 512

    required_pv_columns = ("symbol",)
    required_quote_columns = ("ask_price", "bid_price", "ask_size", "bid_size", "participant_timestamp", "symbol")

    @staticmethod
    def _rth_mask(ts_ns: pd.Series, tz_name: str, start: dt.time, end: dt.time) -> pd.Series:
        ts = pd.to_datetime(ts_ns.astype("Int64"), unit="ns", utc=True)
        et = ts.dt.tz_convert(tz_name)
        h, m = et.dt.hour, et.dt.minute
        ge_start = (h > start.hour) | ((h == start.hour) & (m >= start.minute))
        lt_end   = (h < end.hour)   | ((h == end.hour)   & (m <  end.minute))
        return ge_start & lt_end

    @staticmethod
    def _bin_index(x: np.ndarray, vmax: float, n_bins: int) -> np.ndarray:
        x_clipped = np.clip(x, 0.0, vmax)
        idx = np.floor((x_clipped / vmax) * n_bins).astype(np.int64)
        idx = np.minimum(idx, n_bins - 1)
        return idx

    @staticmethod
    def _pctl_from_hist(counts: np.ndarray, p: float, vmax: float) -> float:
        total = counts.sum()
        if total <= 0:
            return np.nan
        k = int(np.ceil(p * total))
        csum = counts.cumsum()
        bin_idx = int(np.searchsorted(csum, k, side="left"))
        return vmax * (bin_idx + 1) / len(counts)

    def _pass1_build_p90(self, pf: pq.ParquetFile, tz_name: str) -> Dict[str, float]:
        cols = ["symbol", "ask_price", "bid_price", "ask_size", "bid_size", "participant_timestamp"]
        vmax, n_bins = float(self.REL_MAX), int(self.N_BINS)

        hist_by: Dict[str, np.ndarray] = {}

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            rth = self._rth_mask(df["participant_timestamp"], tz_name, self.RTH_START, self.RTH_END)

            a   = pd.to_numeric(df["ask_price"], errors="coerce")
            b   = pd.to_numeric(df["bid_price"], errors="coerce")
            asz = pd.to_numeric(df["ask_size"], errors="coerce")
            bsz = pd.to_numeric(df["bid_size"], errors="coerce")

            denom = a + b
            size_sum = asz + bsz
            valid = (
                rth &
                a.replace([np.inf, -np.inf], np.nan).notna() &
                b.replace([np.inf, -np.inf], np.nan).notna() &
                asz.replace([np.inf, -np.inf], np.nan).notna() &
                bsz.replace([np.inf, -np.inf], np.nan).notna() &
                size_sum.replace([np.inf, -np.inf], np.nan).notna() & (size_sum > 0.0) &
                denom.replace([np.inf, -np.inf], np.nan).notna() & (denom > 0.0)
            )
            if not bool(valid.any()):
                continue

            rel = (2.0 * (a - b) / denom).astype(float).clip(lower=0.0)
            rel = rel.loc[valid].values
            syms = df.loc[valid, "symbol"].astype(str).values

            bin_idx = self._bin_index(rel, vmax, n_bins)
            sym_codes, uniq_syms = pd.factorize(syms, sort=False)
            combined = sym_codes * n_bins + bin_idx
            u, c = np.unique(combined, return_counts=True)
            u_sym = (u // n_bins).astype(np.int64)
            u_bin = (u %  n_bins).astype(np.int64)

            for sidx, bidx, cnt in zip(u_sym, u_bin, c):
                sym = uniq_syms[sidx]
                if sym not in hist_by:
                    hist_by[sym] = np.zeros(n_bins, dtype=np.int64)
                hist_by[sym][bidx] += int(cnt)

        p90_by: Dict[str, float] = {}
        for sym, counts in hist_by.items():
            p90_by[sym] = float(self._pctl_from_hist(counts, 0.90, vmax))
        return p90_by

    def process_date(self, ctx: FeatureContext, date: dt.date):
        sample = self.load_pv(ctx, date, columns=["symbol"])
        if sample is None:
            return None
        if sample.empty:
            return pd.DataFrame(columns=["symbol", "value"])

        root = Path(getattr(ctx, "quote_root", self.default_quote_root))
        day_path = root / f"{date.strftime('%Y%m%d')}.parquet"
        if not day_path.exists():
            out = sample[["symbol"]].copy()
            out["value"] = pd.NA
            return out

        tz_name = getattr(ctx, "tz", "America/New_York")
        pf = pq.ParquetFile(str(day_path))

        p90_by = self._pass1_build_p90(pf, tz_name)

        cols = ["symbol", "ask_price", "bid_price", "ask_size", "bid_size", "participant_timestamp"]
        sum_by: Dict[str, float] = defaultdict(float)
        cnt_by: Dict[str, int] = defaultdict(int)

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()

            rth = self._rth_mask(df["participant_timestamp"], tz_name, self.RTH_START, self.RTH_END)

            a   = pd.to_numeric(df["ask_price"], errors="coerce")
            b   = pd.to_numeric(df["bid_price"], errors="coerce")
            asz = pd.to_numeric(df["ask_size"], errors="coerce")
            bsz = pd.to_numeric(df["bid_size"], errors="coerce")

            denom = a + b
            size_sum = asz + bsz
            valid = (
                rth &
                a.replace([np.inf, -np.inf], np.nan).notna() &
                b.replace([np.inf, -np.inf], np.nan).notna() &
                asz.replace([np.inf, -np.inf], np.nan).notna() &
                bsz.replace([np.inf, -np.inf], np.nan).notna() &
                size_sum.replace([np.inf, -np.inf], np.nan).notna() & (size_sum > 0.0) &
                denom.replace([np.inf, -np.inf], np.nan).notna() & (denom > 0.0)
            )
            if not bool(valid.any()):
                continue

            rel = (2.0 * (a - b) / denom).astype(float).clip(lower=0.0)
            imb = ((bsz - asz) / size_sum).astype(float)

            syms = df.loc[valid, "symbol"].astype(str).values
            relv = rel.loc[valid].values
            imbv = imb.loc[valid].values

            cvals = []
            csyms = []
            for s, r, im in zip(syms, relv, imbv):
                p90 = p90_by.get(s, np.nan)
                if not (np.isfinite(p90) and p90 > 0.0 and np.isfinite(r) and np.isfinite(im)):
                    continue
                factor = 1.0 - (r / p90)
                factor = float(np.clip(factor, 0.0, 1.0))
                cvals.append(im * factor)
                csyms.append(s)

            if not cvals:
                continue

            tmp = pd.DataFrame({"symbol": csyms, "x": np.array(cvals, dtype=float)})
            grp = tmp.groupby("symbol", observed=True)["x"].agg(sum="sum", count="count")
            for k, row in grp.iterrows():
                sum_by[k] += float(row["sum"])
                cnt_by[k] += int(row["count"])

        mean_by = {k: (sum_by[k] / cnt) for k, cnt in cnt_by.items() if cnt > 0}

        out = sample[["symbol"]].copy()
        out["value"] = [mean_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteImbSpreadCouplingOnefileFeature()
