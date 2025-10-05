# feature_base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any
from pathlib import Path
import pandas as pd
import datetime as dt
import os

# ---------- 公共上下文 ----------
@dataclass
class FeatureContext:
    pv_dir: Path                   # 如 ../dataraw/us/pv
    full_dir: Path                 # 如 ../dataraw/us/cubefull
    out_root: Path                 # 如 ../dataraw/us/cube2mat
    tz: str = "America/New_York"   # 交易所本地时区，默认美东
    atomic_write: bool = True
    parquet_compression: Optional[str] = "snappy"

# ---------- 特征基类 ----------
class BaseFeature:
    """
    你需要继承它，并至少定义：
    - name: 输出子目录名（如 'std_close'）
    - process_date(ctx, date) -> pd.DataFrame | None
      返回包含至少 ['symbol', 'value'] 的 DataFrame；None 表示跳过（如缺数据）
    也可按需覆盖：
    - required_full_columns / required_pv_columns
    - output_filename_fmt
    - skip_if_exists: 已有文件是否跳过
    """
    name: str = "base_feature"
    description: str = ""
    required_full_columns: Sequence[str] = ("symbol", "time")
    required_pv_columns: Sequence[str] = ("symbol",)
    output_filename_fmt: str = "{Y}_{m}_{d}.parquet"
    skip_if_exists: bool = True

    # --- 核心：子类必须实现 ---
    def process_date(self, ctx: FeatureContext, date: dt.date) -> Optional[pd.DataFrame]:
        raise NotImplementedError

    # --- 常用工具：子类可用 ---
    def load_full(self, ctx: FeatureContext, date: dt.date, columns: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
        path = ctx.full_dir / f"{date.strftime('%Y%m%d')}.parquet"
        if not path.exists():
            print(f"[{self.name}] miss full: {path}")
            return None
        return pd.read_parquet(path, columns=columns)

    def load_pv(self, ctx: FeatureContext, date: dt.date, columns: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
        path = ctx.pv_dir / f"{date.year}_{date.month}_{date.day}.parquet"
        if not path.exists():
            print(f"[{self.name}] miss pv: {path}")
            return None
        return pd.read_parquet(path, columns=columns)

    def ensure_et_index(self, df: pd.DataFrame, time_col: str = "time", tz: str = "America/New_York") -> pd.DataFrame:
        t = pd.to_datetime(df[time_col], errors="coerce")
        # 没时区 -> 视为本地 tz；有时区 -> 转到 tz
        if getattr(t.dt, "tz", None) is None:
            t = t.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        else:
            t = t.dt.tz_convert(tz)
        return df.set_index(t)

    def out_dir(self, ctx: FeatureContext) -> Path:
        d = ctx.out_root / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def out_path_for_date(self, ctx: FeatureContext, date: dt.date) -> Path:
        fname = self.output_filename_fmt.format(Y=date.year, m=date.month, d=date.day)
        return self.out_dir(ctx) / fname

    def already_done(self, ctx: FeatureContext, date: dt.date) -> bool:
        return self.out_path_for_date(ctx, date).exists()

    def save_output(self, ctx: FeatureContext, date: dt.date, df: pd.DataFrame) -> None:
        out_path = self.out_path_for_date(ctx, date)
        if ctx.atomic_write:
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            df.to_parquet(tmp, index=False, compression=ctx.parquet_compression)
            os.replace(tmp, out_path)
        else:
            df.to_parquet(out_path, index=False, compression=ctx.parquet_compression)

# ---------- 日期工具 ----------
def get_dates_from_pv(pv_dir: Path, start: dt.date, end: dt.date) -> list[dt.date]:
    """
    扫描 pv_dir 中形如 'YYYY_M_D.parquet' 的文件，取交集范围 [start, end]
    """
    dates = []
    for p in pv_dir.glob("*.parquet"):
        stem = p.stem  # e.g., '2018_7_1'
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        try:
            y, m, d = map(int, parts)
            day = dt.date(y, m, d)
        except Exception:
            continue
        if start <= day <= end:
            dates.append(day)
    dates.sort()
    return dates
