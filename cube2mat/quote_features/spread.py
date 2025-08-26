# quote_features/spread.py
from __future__ import annotations
import os, re, glob
import datetime as dt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from feature_base import FeatureContext
from quote_feature_base import QuoteBaseFeature

def _normalize_symbol(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    if " " in s:
        s = s.split(" ")[0]
    s = s.replace("_", ".")
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

class QuoteSpreadAllFeature(QuoteBaseFeature):
    """
    流式 + 自适配版本：
      - 按样本符号逐个读取 {quote_root}/{YYYYMMDD}/{SYMBOL}.parquet
      - 自动匹配文件名与多种候选列名
      - 计算全日平均价差：2*(ask-bid)/(ask+bid)
      - 仅保留聚合结果（内存友好）
    输出：['symbol','value']（按样本顺序对齐）
    """
    name = "quote_spread_all"
    description = "All-day mean of 2*(ask-bid)/(ask+bid) per symbol, streaming & schema-adaptive."
    required_pv_columns    = ("symbol",)
    required_quote_columns = ("participant_timestamp", "bid_price", "ask_price")

    # 扩充：可识别的列名候选（按优先级）
    CANDIDATE_COL_PAIRS: List[Tuple[str, str]] = [
        ("ask_price", "bid_price"),
        ("ask", "bid"),
        ("best_ask", "best_bid"),
        ("best_ask_price", "best_bid_price"),
        ("a1_price", "b1_price"),
        ("askPrice", "bidPrice"),
        ("AskPrice", "BidPrice"),
        ("ASK_PRICE", "BID_PRICE"),
        ("A", "B"),
    ]

    def _build_file_index(self, day_dir: str) -> Tuple[Dict[str, str], Set[str]]:
        index: Dict[str, str] = {}
        stems: Set[str] = set()
        for fp in glob.glob(os.path.join(day_dir, f"*{self.file_ext}")):
            stem = os.path.basename(fp)
            if stem.lower().endswith(self.file_ext):
                stem = stem[:-len(self.file_ext)]
            stems.add(stem)
            norm = _normalize_symbol(stem)
            index.setdefault(norm, stem)
        return index, stems

    def _choose_price_columns(self, path: str) -> Optional[Tuple[str, str]]:
        try:
            import pyarrow.parquet as pq
            names = set(pq.ParquetFile(path).schema.names)
        except Exception:
            try:
                names = set(pd.read_parquet(path, engine="pyarrow").columns)
            except Exception:
                try:
                    names = set(pd.read_parquet(path).columns)
                except Exception:
                    return None
        for a, b in self.CANDIDATE_COL_PAIRS:
            if a in names and b in names:
                return (a, b)
        # 尝试忽略大小写
        lower_map = {x.lower(): x for x in names}
        for a, b in self.CANDIDATE_COL_PAIRS:
            aa, bb = lower_map.get(a.lower()), lower_map.get(b.lower())
            if aa and bb:
                return (aa, bb)
        return None

    def _mean_spread_from_file(self, path: str, col_ask: str, col_bid: str):
        # 1) pyarrow 流式（快速 + 省内存）
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            sum_spread = 0.0
            count = 0
            for rb in pf.iter_batches(columns=[col_ask, col_bid], batch_size=500_000):
                a = pc.cast(rb.column(0), pa.float64())
                b = pc.cast(rb.column(1), pa.float64())
                denom = pc.add(a, b)
                valid = pc.and_(pc.is_finite(a), pc.is_finite(b))
                valid = pc.and_(valid, pc.is_finite(denom))
                valid = pc.and_(valid, pc.greater(denom, pc.scalar(0.0)))
                two_diff = pc.multiply(pc.subtract(a, b), pc.scalar(2.0))
                spread = pc.divide_checked(two_diff, denom)
                spread_valid = pc.filter(spread, valid)
                if spread_valid.length() == 0:
                    continue
                s = pc.sum(spread_valid, skip_nulls=True).as_py()
                c = pc.count(spread_valid, mode="only_valid").as_py()
                if s is not None and c:
                    sum_spread += float(s); count += int(c)
            if count == 0:
                return pd.NA
            return sum_spread / count
        except Exception:
            pass

        # 2) 回退：pandas 只读两列
        try:
            df = pd.read_parquet(path, columns=[col_ask, col_bid])
        except Exception:
            return pd.NA
        df = df.copy()
        df[col_ask] = pd.to_numeric(df[col_ask], errors="coerce")
        df[col_bid] = pd.to_numeric(df[col_bid], errors="coerce")
        df = df.dropna(subset=[col_ask, col_bid])
        if df.empty:
            return pd.NA
        denom = df[col_ask] + df[col_bid]
        spread = 2.0 * (df[col_ask] - df[col_bid]) / denom
        valid = np.isfinite(spread) & np.isfinite(denom) & (denom > 0)
        s = spread[valid]
        if s.empty:
            return pd.NA
        return float(s.mean())

    def process_date(self, ctx: FeatureContext, date: dt.date):
        sample = self.load_pv(ctx, date, columns=["symbol"])
        if sample is None:
            return None
        if sample.empty:
            return pd.DataFrame(columns=["symbol", "value"])

        day_dir = self.quote_day_dir(ctx, date)
        file_index, _ = self._build_file_index(day_dir)

        results: Dict[str, float] = {}
        for sym in pd.Series(sample["symbol"]).astype(str):
            norm = _normalize_symbol(sym)
            stem = file_index.get(norm)
            if stem is None and "." in sym:
                stem = file_index.get(_normalize_symbol(sym.split(".")[0]))
            if stem is None and "-" in sym:
                stem = file_index.get(_normalize_symbol(sym.split("-")[0]))
            if stem is None:
                results[sym] = pd.NA
                continue

            path = os.path.join(day_dir, f"{stem}{self.file_ext}")
            if not os.path.exists(path):
                results[sym] = pd.NA
                continue

            cols = self._choose_price_columns(path)
            if cols is None:
                results[sym] = pd.NA
                continue

            col_ask, col_bid = cols
            try:
                results[sym] = self._mean_spread_from_file(path, col_ask, col_bid)
            except Exception:
                results[sym] = pd.NA

        out = sample[["symbol"]].copy()
        out["value"] = out["symbol"].astype(str).map(results)
        return out

feature = QuoteSpreadAllFeature()
