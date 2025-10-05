# quote_features/nonfirm_manual_share.py
from __future__ import annotations
import datetime as dt
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, Tuple, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature, DATARAW_ROOT


def _has_any_code(x: Iterable, code_set: Set[int]) -> bool:
    if not code_set or x is None:
        return False
    try:
        for v in x:
            if v in code_set:
                return True
    except TypeError:
        # 非可迭代或 NA
        return False
    return False


class QuoteNonFirmManualShareOnefileFeature(QuoteBaseFeature):
    """
    在 RTH(09:30–16:00 ET) 内，按事件统计 “非有效 / 手工” 报价占比：
      share = #( has_nonfirm_or_manual ) / #( all RTH events )
    其中 has_nonfirm_or_manual 由 conditions/indicators 是否包含指定代码决定。
    代码集合从 ctx 注入（均为 set[int]；默认为空集合）：
      - nonfirm_condition_codes / nonfirm_indicator_codes
      - manual_condition_codes  / manual_indicator_codes
    输出：['symbol','value']，事件占比 ∈ [0,1]。
    """
    name = "quote_nonfirm_manual_share_all"
    description = "RTH event share of non-firm or manual quotes based on conditions/indicators (onefile)"
    default_quote_root = str(DATARAW_ROOT / "us" / "quote_onefile")

    RTH_START = dt.time(9, 30)
    RTH_END   = dt.time(16, 0)
    BATCH_SIZE = 500_000

    required_pv_columns = ("symbol",)
    required_quote_columns = ("conditions", "indicators", "participant_timestamp", "symbol")

    @staticmethod
    def _rth_mask(ts_ns: pd.Series, tz_name: str, start: dt.time, end: dt.time) -> pd.Series:
        ts = pd.to_datetime(ts_ns.astype("Int64"), unit="ns", utc=True)
        et = ts.dt.tz_convert(tz_name)
        h, m = et.dt.hour, et.dt.minute
        ge_start = (h > start.hour) | ((h == start.hour) & (m >= start.minute))
        lt_end   = (h < end.hour)   | ((h == end.hour)   & (m <  end.minute))
        return ge_start & lt_end

    def process_date(self, ctx: FeatureContext, date: dt.date):
        # 1) PV
        sample = self.load_pv(ctx, date, columns=["symbol"])
        if sample is None:
            return None
        if sample.empty:
            return pd.DataFrame(columns=["symbol", "value"])

        # 2) 文件
        root = Path(getattr(ctx, "quote_root", self.default_quote_root))
        path = root / f"{date.strftime('%Y%m%d')}.parquet"
        if not path.exists():
            out = sample[["symbol"]].copy(); out["value"] = pd.NA; return out

        # 代码集合
        nf_c = set(getattr(ctx, "nonfirm_condition_codes", set()))
        nf_i = set(getattr(ctx, "nonfirm_indicator_codes", set()))
        ma_c = set(getattr(ctx, "manual_condition_codes", set()))
        ma_i = set(getattr(ctx, "manual_indicator_codes", set()))

        tz_name = getattr(ctx, "tz", "America/New_York")
        pf = pq.ParquetFile(str(path))
        cols = ["symbol", "conditions", "indicators", "participant_timestamp"]

        num_by: Dict[str, int] = defaultdict(int)
        den_by: Dict[str, int] = defaultdict(int)

        for rb in pf.iter_batches(columns=cols, batch_size=self.BATCH_SIZE):
            df = rb.to_pandas()
            ts = pd.to_numeric(df["participant_timestamp"], errors="coerce")
            rth = self._rth_mask(ts, tz_name, self.RTH_START, self.RTH_END)
            if not bool(rth.any()):
                continue

            sub = pd.DataFrame({
                "symbol": df.loc[rth, "symbol"].astype(str).values,
                "conditions": df.loc[rth, "conditions"],
                "indicators": df.loc[rth, "indicators"],
            })

            # 事件是否命中 non-firm / manual 标志
            nonfirm = sub["conditions"].apply(lambda x: _has_any_code(x, nf_c)) | \
                      sub["indicators"].apply(lambda x: _has_any_code(x, nf_i))
            manual  = sub["conditions"].apply(lambda x: _has_any_code(x, ma_c)) | \
                      sub["indicators"].apply(lambda x: _has_any_code(x, ma_i))
            bad = (nonfirm | manual).astype(int)

            tmp = pd.DataFrame({"symbol": sub["symbol"].values, "den": 1, "num": bad.values}) \
                    .groupby("symbol", observed=True).sum()

            for sym, row in tmp.iterrows():
                den_by[sym] += int(row["den"])
                num_by[sym] += int(row["num"])

        share_by = {k: (num_by[k] / den) for k, den in den_by.items() if den > 0}

        out = sample[["symbol"]].copy()
        out["value"] = [share_by.get(str(s), pd.NA) for s in sample["symbol"]]
        return out


feature = QuoteNonFirmManualShareOnefileFeature()
