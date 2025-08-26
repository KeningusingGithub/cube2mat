# quote_feature_base.py
from __future__ import annotations
import os
import glob
import datetime as dt
from typing import Iterable, Optional, Sequence, Dict, List, Set
import pandas as pd
from feature_base import BaseFeature, FeatureContext

class QuoteBaseFeature(BaseFeature):
    """
    面向 quote 源的数据基类（已增强以自适配列名差异）：
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

    # 新增：可识别的“时间戳列”候选（按优先级）
    TIME_COL_CANDIDATES: Sequence[str] = (
        "participant_timestamp", "ParticipantTimestamp", "participant_ts",
        "sip_timestamp", "SIPTimestamp", "sip_ts",
        "timestamp", "time", "ts", "T"
    )

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

    # --- schema 辅助 ---
    def _get_schema_names(self, path: str) -> Optional[Set[str]]:
        try:
            import pyarrow.parquet as pq
            return set(pq.ParquetFile(path).schema.names)
        except Exception:
            try:
                # 回退：pandas 读取一次列名（不指定列）
                return set(pd.read_parquet(path).columns)
            except Exception:
                return None

    def _choose_time_column(self, names: Set[str]) -> Optional[str]:
        if not names:
            return None
        # 原样匹配
        for n in self.TIME_COL_CANDIDATES:
            if n in names:
                return n
        # 忽略大小写匹配
        lower_map = {x.lower(): x for x in names}
        for n in self.TIME_COL_CANDIDATES:
            k = n.lower()
            if k in lower_map:
                return lower_map[k]
        return None

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
        变更：自动探测并标准化时间戳列为 'participant_timestamp'。
        """
        files = self.list_quote_files(ctx, date, symbols)
        want_cols = set(columns or self.required_quote_columns)
        # 统一确保 'participant_timestamp' 在输出里
        want_cols.add("participant_timestamp")

        if not files:
            # 返回带齐列名的空表
            cols = ["symbol", *list(want_cols)]
            return pd.DataFrame(columns=cols)

        dfs: List[pd.DataFrame] = []
        for path in files:
            names = self._get_schema_names(path)
            if not names:
                continue

            time_col = self._choose_time_column(names)
            # 需要读取的列：去掉 'participant_timestamp' 占位，改读实际 time_col
            to_read = set(want_cols) - {"participant_timestamp"}
            # 只读文件里实际存在的列
            cols_to_read = [c for c in to_read if c in names]
            # 一定把 time_col 也加入（若找到）
            if time_col is not None and time_col not in cols_to_read:
                cols_to_read.append(time_col)

            try:
                if cols_to_read:
                    df = pd.read_parquet(path, columns=cols_to_read)
                else:
                    # 没有任何字段可直读，退化为全读
                    df = pd.read_parquet(path)
            except Exception:
                # 容错：全读
                try:
                    df = pd.read_parquet(path)
                except Exception:
                    continue

            if df is None or df.empty:
                continue

            df = df.copy()

            # 统一生成 'participant_timestamp'
            if "participant_timestamp" in df.columns:
                # 已经是目标名，直接标准化为 Int64
                df["participant_timestamp"] = pd.to_numeric(
                    df["participant_timestamp"], errors="coerce"
                ).astype("Int64")
            else:
                # 用探测到的 time_col 映射
                if time_col and time_col in df.columns:
                    # 先尽量走数值路径（ns）
                    ser = pd.to_numeric(df[time_col], errors="coerce")
                    if ser.notna().any():
                        df["participant_timestamp"] = ser.astype("Int64")
                    else:
                        # 回退：当作 datetime 解析再转 ns
                        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
                        # ts.value 是 int ns；但 series.dt.value 需先 dropna
                        df["participant_timestamp"] = ts.view("int64").astype("Int64")
                else:
                    # 找不到时间列：留空以便后续 dropna 过滤
                    df["participant_timestamp"] = pd.NA

            # 补 symbol（来自文件名）
            df.insert(0, "symbol", self._file_symbol(path))

            # 去掉无效时间戳
            df = df.dropna(subset=["participant_timestamp"])

            # 去掉“业务列全为 NaN”的行
            data_cols = [c for c in df.columns if c not in ("symbol", "participant_timestamp")]
            if data_cols:
                df = df.dropna(how="all", subset=data_cols)

            if not df.empty:
                # 只保留需要的输出列
                keep = ["symbol", "participant_timestamp", *(c for c in (columns or []) if c in df.columns)]
                # 避免重复
                keep = list(dict.fromkeys(keep))
                df = df[keep]
                dfs.append(df)

        if not dfs:
            cols = ["symbol", *list(want_cols)]
            return pd.DataFrame(columns=cols)

        if len(dfs) == 1:
            return dfs[0].reset_index(drop=True)

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
        ts = pd.to_datetime(df["participant_timestamp"].astype("Int64"), unit="ns", utc=True)
        ts = ts.dt.tz_convert(tz)
        df = df.set_index(ts)
        return df
