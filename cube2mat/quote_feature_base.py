# quote_feature_base.py
from __future__ import annotations
import os
import glob
import datetime as dt
from typing import Iterable, Optional, Sequence, Dict, List
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class QuoteBaseFeature(BaseFeature):
    """
    面向 quote 源的数据基类：
    - 根据日期扫描 {quote_root}/{YYYYMMDD}/*.parquet
    - 可按 symbols 子集加载
    - 自动补充 'symbol' 列（来自文件名）
    - 统一 participant_timestamp 为 int ns，并可转换为时区索引
    """

    # 默认根目录，可被 ctx.quote_root 覆盖
    default_quote_root = "/home/ubuntu/dataraw/us/quote"

    # 子类可覆盖：文件扩展名
    file_ext = ".parquet"

    # 子类必须在 required_quote_columns 中至少包含 'participant_timestamp'
    required_quote_columns: Sequence[str] = ("participant_timestamp",)

    def quote_day_dir(self, ctx: FeatureContext, date: dt.date) -> str:
        root = getattr(ctx, "quote_root", self.default_quote_root)
        return os.path.join(root, date.strftime("%Y%m%d"))

    def _file_symbol(self, path: str) -> str:
        """从文件名提取 symbol，如 'AAC.parquet' -> 'AAC'"""
        base = os.path.basename(path)
        if base.lower().endswith(self.file_ext):
            base = base[: -len(self.file_ext)]
        return base

    def list_quote_files(self, ctx: FeatureContext, date: dt.date,
                         symbols: Optional[Iterable[str]] = None) -> List[str]:
        day_dir = self.quote_day_dir(ctx, date)
        if symbols:
            files = [os.path.join(day_dir, f"{sym}{self.file_ext}") for sym in symbols]
            return [f for f in files if os.path.exists(f)]
        # 否则全量扫描
        return sorted(glob.glob(os.path.join(day_dir, f"*{self.file_ext}")))

    def load_quote_full(
        self,
        ctx: FeatureContext,
        date: dt.date,
        symbols: Optional[Iterable[str]] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        加载当日 quote 全量/子集（按 symbols），返回统一结构：
        ['symbol', <columns...>]，其中 participant_timestamp 为 ns（pandas Int64）。
        """
        files = self.list_quote_files(ctx, date, symbols)
        want_cols = set(columns or self.required_quote_columns)
        # 确保必要列
        want_cols.add("participant_timestamp")

        if not files:
            # 返回带齐列名的空表
            cols = ["symbol", *list(want_cols)]
            return pd.DataFrame(columns=cols)

        dfs: List[pd.DataFrame] = []
        want_cols_list = list(want_cols)

        for path in files:
            try:
                # 优先按列读取
                df = pd.read_parquet(path, columns=want_cols_list)
            except Exception:
                # 容错：全读再补列
                df = pd.read_parquet(path)
                # 补齐缺失列为 NA
                missing = want_cols - set(df.columns)
                for c in missing:
                    df[c] = pd.NA
                df = df[list(want_cols)]  # 统一列顺序

            if df.empty:
                continue

            df = df.copy()
            df.insert(0, "symbol", self._file_symbol(path))

            # 时间戳转 Int64 ns，并去掉无效时间戳
            df["participant_timestamp"] = pd.to_numeric(
                df["participant_timestamp"], errors="coerce"
            ).astype("Int64")
            df = df.dropna(subset=["participant_timestamp"])

            # 去掉“业务列全为 NaN”的行（避免 all-NA 帧进入 concat）
            data_cols = [c for c in df.columns if c not in ("symbol", "participant_timestamp")]
            if data_cols:
                df = df.dropna(how="all", subset=data_cols)

            if not df.empty:
                dfs.append(df)

        if not dfs:
            # 仍可能因为清洗后全空；返回规范空表
            cols = ["symbol", *list(want_cols)]
            return pd.DataFrame(columns=cols)

        # 只有一个条目时直接返回，避免 concat 的 FutureWarning
        if len(dfs) == 1:
            return dfs[0].reset_index(drop=True)

        # 多个条目再拼接；因为已排除 empty / all-NA，不会触发 FutureWarning
        out = pd.concat(dfs, axis=0, ignore_index=True, copy=False)
        return out


    # ---- 时间辅助 ----
    @staticmethod
    def datetime_to_ns(x, tz) -> Optional[int]:
        """
        把样本中的 time 转换为 UTC ns：
        - 若 naive，则假定在交易所时区 tz，再转 UTC。
        - 若 aware，则直接转 UTC。
        - 若为 str，先 parse 再处理。
        """
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        ts = pd.to_datetime(x, errors="coerce")
        if ts is pd.NaT:
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        ts_utc = ts.tz_convert("UTC")
        return int(ts_utc.value)  # ns since epoch

    def ensure_participant_index(self, df: pd.DataFrame, tz) -> pd.DataFrame:
        """
        用 participant_timestamp 建立索引并转交易所时区（便于 between_time 等操作）
        """
        df = df.copy()
        # 转为 UTC 再到 tz
        ts = pd.to_datetime(df["participant_timestamp"].astype("Int64"), unit="ns", utc=True)
        ts = ts.dt.tz_convert(tz)
        df = df.set_index(ts)
        return df
