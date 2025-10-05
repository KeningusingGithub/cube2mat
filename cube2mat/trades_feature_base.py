# trades_feature_base.py
from __future__ import annotations

import os
import glob
import datetime as dt
from functools import partial
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Set, Union

import pandas as pd

try:  # pragma: no cover - optional dependency
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - pyarrow 可能未安装
    pa = None
    pq = None

from feature_base import BaseFeature, FeatureContext


DATARAW_ROOT = Path(__file__).resolve().parents[1].parent / "dataraw"


# ---- runner 注入的 helpers -------------------------------------------------
def _trade_day_dir(ctx: FeatureContext, date: dt.date, *, file_ext: str) -> Path:
    root = Path(getattr(ctx, "trade_root", str(DATARAW_ROOT / "us" / "trade")))
    return root / date.strftime("%Y%m%d")


def _trade_day_file(ctx: FeatureContext, date: dt.date, *, file_ext: str) -> Path:
    root = Path(getattr(ctx, "trade_root", str(DATARAW_ROOT / "us" / "trade")))
    return root / f"{date.strftime('%Y%m%d')}{file_ext}"


def _resolve_layout(ctx: FeatureContext, date: dt.date, *, file_ext: str) -> str:
    """返回 'directory' | 'onefile' | 'missing'。"""

    prefer = getattr(ctx, "trade_layout", "auto")
    day_dir = _trade_day_dir(ctx, date, file_ext=file_ext)
    day_file = _trade_day_file(ctx, date, file_ext=file_ext)

    if isinstance(prefer, str) and prefer.lower() in ("dir", "directory"):
        return "directory" if day_dir.exists() else "missing"
    if isinstance(prefer, str) and prefer.lower() in ("onefile", "file", "single"):
        return "onefile" if day_file.exists() else "missing"

    if day_dir.exists():
        return "directory"
    if day_file.exists():
        return "onefile"
    return "missing"


def attach_trade_helpers(
    ctx: FeatureContext,
    *,
    trade_root: Optional[Union[str, Path]] = None,
    trade_layout: str = "auto",
    file_ext: str = ".parquet",
) -> None:
    """
    在 runner 中调用，把 trades 的读数 helper 注入到 FeatureContext。
    """

    if trade_root is not None:
        setattr(ctx, "trade_root", str(trade_root))
    setattr(ctx, "trade_layout", trade_layout)
    setattr(ctx, "trade_file_ext", file_ext)

    ctx.iterate_day_in_batches = partial(_iterate_day_in_batches_ctx, ctx)
    ctx.iterate_day_for_symbol = partial(_iterate_day_for_symbol_ctx, ctx)


def _iterate_day_in_batches_ctx(
    ctx: FeatureContext,
    date: dt.date,
    *,
    columns: Optional[Sequence[str]] = None,
    symbols: Optional[Iterable[str]] = None,
    batch_size: int = 1_000_000,
) -> Generator[pd.DataFrame, None, None]:
    file_ext = getattr(ctx, "trade_file_ext", ".parquet")
    layout = _resolve_layout(ctx, date, file_ext=file_ext)
    if layout == "missing":
        return
        yield  # pragma: no cover - 维持生成器签名

    if layout == "directory":
        day_dir = _trade_day_dir(ctx, date, file_ext=file_ext)
        wanted_syms = set(str(s) for s in symbols) if symbols else None
        files = sorted(glob.glob(os.path.join(str(day_dir), f"*{file_ext}")))
        for path in files:
            file_sym = Path(path).stem
            if wanted_syms is not None and file_sym not in wanted_syms:
                continue
            yield from _yield_parquet_batches(
                path,
                columns=columns,
                symbol_from_filename=file_sym,
                batch_size=batch_size,
            )
    else:
        day_file = _trade_day_file(ctx, date, file_ext=file_ext)
        yield from _yield_parquet_batches(
            str(day_file),
            columns=columns,
            symbol_filter=set(str(s) for s in symbols) if symbols else None,
            batch_size=batch_size,
        )


def _iterate_day_for_symbol_ctx(
    ctx: FeatureContext,
    date: dt.date,
    symbol: str,
    *,
    columns: Optional[Sequence[str]] = None,
    batch_size: int = 1_000_000,
) -> Generator[pd.DataFrame, None, None]:
    file_ext = getattr(ctx, "trade_file_ext", ".parquet")
    layout = _resolve_layout(ctx, date, file_ext=file_ext)
    if layout == "missing":
        return
        yield  # pragma: no cover - 维持生成器签名

    if layout == "directory":
        day_dir = _trade_day_dir(ctx, date, file_ext=file_ext)
        path = day_dir / f"{symbol}{file_ext}"
        if path.exists():
            yield from _yield_parquet_batches(
                str(path),
                columns=columns,
                symbol_from_filename=str(symbol),
                batch_size=batch_size,
            )
    else:
        day_file = _trade_day_file(ctx, date, file_ext=file_ext)
        yield from _yield_parquet_batches(
            str(day_file),
            columns=columns,
            symbol_filter={str(symbol)},
            batch_size=batch_size,
        )


# ---- parquet 读取 + 时间戳标准化 -------------------------------------------
_TIME_COL_CANDIDATES: Sequence[str] = (
    "participant_timestamp",
    "ParticipantTimestamp",
    "participant_ts",
    "trade_timestamp",
    "TradeTimestamp",
    "trade_ts",
    "sip_timestamp",
    "SIPTimestamp",
    "sip_ts",
    "timestamp",
    "time",
    "ts",
    "T",
)


def _read_schema_names(path: str) -> Optional[Set[str]]:
    try:  # pragma: no cover - 依赖 pyarrow
        if pq is not None:
            return set(pq.ParquetFile(path).schema.names)
    except Exception:
        pass

    try:
        return set(pd.read_parquet(path).columns)
    except Exception:
        return None


def _choose_time_column(names: Set[str]) -> Optional[str]:
    if not names:
        return None
    for n in _TIME_COL_CANDIDATES:
        if n in names:
            return n
    lower_map = {x.lower(): x for x in names}
    for n in _TIME_COL_CANDIDATES:
        key = n.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _standardize_participant_ts_inplace(df: pd.DataFrame, time_col: Optional[str]) -> None:
    if "participant_timestamp" in df.columns:
        df["participant_timestamp"] = pd.to_numeric(
            df["participant_timestamp"], errors="coerce"
        ).astype("Int64")
        return

    if time_col and time_col in df.columns:
        ser = pd.to_numeric(df[time_col], errors="coerce")
        if ser.notna().any():
            df["participant_timestamp"] = ser.astype("Int64")
            return

        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df["participant_timestamp"] = ts.view("int64").astype("Int64")
        return

    df["participant_timestamp"] = pd.NA


def _yield_parquet_batches(
    path: str,
    *,
    columns: Optional[Sequence[str]],
    symbol_from_filename: Optional[str] = None,
    symbol_filter: Optional[Set[str]] = None,
    batch_size: int = 1_000_000,
) -> Generator[pd.DataFrame, None, None]:
    want_cols = set(columns or {"participant_timestamp"})
    want_cols.add("participant_timestamp")

    names = _read_schema_names(path) or set()
    time_col = _choose_time_column(names)

    to_read = list((want_cols - {"participant_timestamp"}) & names)
    if time_col and time_col not in to_read:
        to_read.append(time_col)

    if "symbol" in want_cols and "symbol" in names and "symbol" not in to_read:
        to_read.append("symbol")

    if pq is None:
        df0 = pd.read_parquet(path, columns=to_read or None)
        if df0 is None or df0.empty:
            return
            yield  # pragma: no cover - 保持生成器

        df0 = df0.copy()
        _standardize_participant_ts_inplace(df0, time_col)

        if "symbol" not in df0.columns and symbol_from_filename is not None:
            df0.insert(0, "symbol", str(symbol_from_filename))

        if symbol_filter is not None and "symbol" in df0.columns:
            df0 = df0[df0["symbol"].astype(str).isin(symbol_filter)]

        df0 = df0.dropna(subset=["participant_timestamp"])
        data_cols = [c for c in df0.columns if c not in ("symbol", "participant_timestamp")]
        if data_cols:
            df0 = df0.dropna(how="all", subset=data_cols)
        if df0.empty:
            return

        keep = [
            "symbol" if "symbol" in df0.columns else None,
            "participant_timestamp",
            *[c for c in (columns or []) if c in df0.columns],
        ]
        keep = [c for c in keep if c is not None]
        if keep:
            yield df0[keep]
        else:
            yield df0
        return

    pf = pq.ParquetFile(path)
    iter_kwargs: Dict[str, Union[int, Sequence[str]]] = {"batch_size": max(1, int(batch_size))}
    if to_read:
        iter_kwargs["columns"] = to_read

    for batch in pf.iter_batches(**iter_kwargs):
        tb = pa.Table.from_batches([batch])
        pdf = tb.to_pandas(ignore_metadata=True)

        _standardize_participant_ts_inplace(pdf, time_col)

        if "symbol" not in pdf.columns and symbol_from_filename is not None:
            pdf.insert(0, "symbol", str(symbol_from_filename))

        if symbol_filter is not None and "symbol" in pdf.columns:
            pdf = pdf[pdf["symbol"].astype(str).isin(symbol_filter)]

        if pdf is None or pdf.empty:
            continue

        pdf = pdf.dropna(subset=["participant_timestamp"])
        data_cols = [c for c in pdf.columns if c not in ("symbol", "participant_timestamp")]
        if data_cols:
            pdf = pdf.dropna(how="all", subset=data_cols)
        if pdf.empty:
            continue

        keep = [
            "symbol" if "symbol" in pdf.columns else None,
            "participant_timestamp",
            *[c for c in (columns or []) if c in pdf.columns],
        ]
        keep = [c for c in keep if c is not None]
        yield pdf[keep] if keep else pdf


# ---- 基类 -------------------------------------------------------------------
class TradesBaseFeature(BaseFeature):
    """
    面向 trades 源的特征基类：
    - 根据日期扫描目录式 {trade_root}/{YYYYMMDD}/*.parquet 或 onefile {trade_root}/{YYYYMMDD}.parquet
    - 可按 symbols 子集加载
    - 自动补 'symbol' 列
    - 统一 participant_timestamp 为 pandas Int64 ns
    """

    default_trade_root = str(DATARAW_ROOT / "us" / "trade")
    file_ext = ".parquet"
    required_trade_columns: Sequence[str] = ("participant_timestamp",)
    TIME_COL_CANDIDATES: Sequence[str] = _TIME_COL_CANDIDATES

    def trade_day_dir(self, ctx: FeatureContext, date: dt.date) -> str:
        root = getattr(ctx, "trade_root", self.default_trade_root)
        return os.path.join(root, date.strftime("%Y%m%d"))

    def trade_day_file(self, ctx: FeatureContext, date: dt.date) -> str:
        root = getattr(ctx, "trade_root", self.default_trade_root)
        return os.path.join(root, f"{date.strftime('%Y%m%d')}{self.file_ext}")

    def _file_symbol(self, path: str) -> str:
        base = os.path.basename(path)
        if base.lower().endswith(self.file_ext):
            base = base[: -len(self.file_ext)]
        return base

    def list_trade_files(
        self,
        ctx: FeatureContext,
        date: dt.date,
        symbols: Optional[Iterable[str]] = None,
    ) -> List[str]:
        day_dir = self.trade_day_dir(ctx, date)
        if symbols:
            files = [os.path.join(day_dir, f"{sym}{self.file_ext}") for sym in symbols]
            return [f for f in files if os.path.exists(f)]
        return sorted(glob.glob(os.path.join(day_dir, f"*{self.file_ext}")))

    def _get_schema_names(self, path: str) -> Optional[Set[str]]:
        return _read_schema_names(path)

    def _choose_time_column(self, names: Set[str]) -> Optional[str]:
        return _choose_time_column(names)

    def load_trades_full(
        self,
        ctx: FeatureContext,
        date: dt.date,
        symbols: Optional[Iterable[str]] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> Optional[pd.DataFrame]:
        batches = list(
            self.iterate_trade_batches(
                ctx,
                date,
                columns=columns,
                symbols=symbols,
            )
        )
        if not batches:
            cols = ["symbol", "participant_timestamp", *(columns or [])]
            return pd.DataFrame(columns=cols)
        out = pd.concat(batches, axis=0, ignore_index=True, copy=False)
        return out

    def iterate_trade_batches(
        self,
        ctx: FeatureContext,
        date: dt.date,
        *,
        columns: Optional[Sequence[str]] = None,
        symbols: Optional[Iterable[str]] = None,
        batch_size: int = 1_000_000,
    ) -> Generator[pd.DataFrame, None, None]:
        if hasattr(ctx, "iterate_day_in_batches"):
            yield from ctx.iterate_day_in_batches(
                date,
                columns=columns,
                symbols=symbols,
                batch_size=batch_size,
            )
            return

        file_ext = getattr(ctx, "trade_file_ext", self.file_ext)
        layout = _resolve_layout(ctx, date, file_ext=file_ext)
        if layout == "missing":
            return
            yield  # pragma: no cover - 维持生成器

        if layout == "directory":
            files = self.list_trade_files(ctx, date, symbols=symbols)
            for path in files:
                file_sym = self._file_symbol(path)
                yield from _yield_parquet_batches(
                    path,
                    columns=columns,
                    symbol_from_filename=file_sym,
                    batch_size=batch_size,
                )
        else:
            day_file = self.trade_day_file(ctx, date)
            yield from _yield_parquet_batches(
                day_file,
                columns=columns,
                symbol_filter=set(str(s) for s in symbols) if symbols else None,
                batch_size=batch_size,
            )

    @staticmethod
    def datetime_to_ns(x, tz) -> Optional[int]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        ts = pd.to_datetime(x, errors="coerce")
        if ts is pd.NaT:
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        ts_utc = ts.tz_convert("UTC")
        return int(ts_utc.value)

    def ensure_participant_index(self, df: pd.DataFrame, tz) -> pd.DataFrame:
        df = df.copy()
        ts = pd.to_datetime(df["participant_timestamp"].astype("Int64"), unit="ns", utc=True)
        ts = ts.dt.tz_convert(tz)
        return df.set_index(ts)

    def get_trade_path(
        self,
        ctx: FeatureContext,
        date: dt.date,
        trade_root: Optional[str] = None,
    ) -> Optional[Path]:
        if trade_root is not None:
            ctx.trade_root = trade_root
        file_ext = getattr(ctx, "trade_file_ext", self.file_ext)
        layout = _resolve_layout(ctx, date, file_ext=file_ext)
        if layout == "missing":
            return None
        if layout == "directory":
            return _trade_day_dir(ctx, date, file_ext=file_ext)
        return _trade_day_file(ctx, date, file_ext=file_ext)


__all__ = [
    "TradesBaseFeature",
    "DATARAW_ROOT",
    "attach_trade_helpers",
]

