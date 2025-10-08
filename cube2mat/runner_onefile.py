#!/usr/bin/env python3
"""Run features into a single Parquet file per feature."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import traceback
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from feature_base import BaseFeature, FeatureContext, get_dates_from_pv


# ---------- 动态加载 feature 模块（与 runner.py 保持一致的语义） ----------
def load_feature_from_file(pyfile: Path) -> Optional[BaseFeature]:
    """Load a feature instance from the given python file."""

    try:
        spec = spec_from_file_location(pyfile.stem, pyfile)
        if spec is None or spec.loader is None:
            print(f"[load] cannot load spec: {pyfile}")
            return None
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]

        if hasattr(mod, "feature"):
            feat = getattr(mod, "feature")
            if isinstance(feat, BaseFeature):
                return feat
            print(f"[load] {pyfile} 'feature' 不是 BaseFeature 实例")
            return None

        candidates = [
            v for v in mod.__dict__.values()
            if isinstance(v, type) and issubclass(v, BaseFeature) and v is not BaseFeature
        ]
        if len(candidates) == 1:
            return candidates[0]()
        if len(candidates) > 1:
            print(f"[load] {pyfile} 有多个 BaseFeature 子类，请显式提供 `feature = Class()`")
        else:
            print(f"[load] {pyfile} 未发现特征类或 `feature` 变量")
        return None
    except Exception:
        print(f"[load] error while importing {pyfile}:\n{traceback.format_exc()}")
        return None


def discover_feature_files(
    feat_dir: Path,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> list[Path]:
    include_set = set(include or [])
    exclude_set = set(exclude or [])
    files: list[Path] = []
    for py in sorted(feat_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        if include_set and py.stem not in include_set:
            continue
        if py.stem in exclude_set:
            continue
        files.append(py)
    return files


# ---------- 数据规范化：确保 ['time','symbol','value'] ----------
def _normalize_for_write(df: pd.DataFrame, date: dt.date) -> pd.DataFrame:
    """Normalize feature output to a consistent schema."""

    out = df.copy()
    if "time" not in out.columns:
        out.insert(0, "time", pd.Timestamp(date))

    cols = [c for c in ["time", "symbol", "value"] if c in out.columns]
    out = out[cols]

    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype(float)
    return out


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    dataraw_root = base_dir.parent.parent.parent / "dataraw"
    parser = argparse.ArgumentParser(
        description="Feature runner (one-file): 按日期计算并写入单一 Parquet 文件",
    )
    parser.add_argument("--pv-dir", default=str(dataraw_root / "us" / "basedata" / "close"))
    parser.add_argument("--full-dir", default=str(dataraw_root / "us" / "cubefull"))
    parser.add_argument("--out-root", default=str(dataraw_root / "us" / "cube2mat"))
    parser.add_argument("--tz", default="America/New_York")
    parser.add_argument("--start", default="2018-07-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD，默认= cubefull 目录最新日期")
    parser.add_argument("--features-dir", default=str(base_dir / "features"))
    parser.add_argument("--only", nargs="*", help="只运行这些特征（文件名不带 .py）")
    parser.add_argument("--exclude", nargs="*", help="排除这些特征")
    parser.add_argument("--single-file-name", default="_single.parquet", help="输出文件名（每个特征目录下）")
    parser.add_argument("--overwrite", action="store_true", help="若单文件存在则覆盖重写")
    return parser.parse_args()


# ---------- 主逻辑 ----------
def main() -> None:
    args = parse_args()
    pv_dir = Path(args.pv_dir)
    full_dir = Path(args.full_dir)
    out_root = Path(args.out_root)
    feat_dir = Path(args.features_dir)

    if args.end is None:
        latest = max((p.stem for p in full_dir.glob("*.parquet")), default=None)
        if latest is None:
            raise SystemExit("cubefull 目录为空，无法推断 end 日期")
        end_date = dt.datetime.strptime(latest, "%Y%m%d").date()
    else:
        end_date = dt.date.fromisoformat(args.end)
    start_date = dt.date.fromisoformat(args.start)

    dates = get_dates_from_pv(pv_dir, start_date, end_date)
    if not dates:
        raise SystemExit("未发现任何有效日期（检查 pv_dir / 起止日期）")

    feat_files = discover_feature_files(feat_dir, include=args.only, exclude=args.exclude)
    if not feat_files:
        raise SystemExit("features 目录下没有可运行的 .py 特征文件")

    for py in feat_files:
        feature = load_feature_from_file(py)
        if feature is None:
            print(f"[runner_onefile] 跳过（加载失败）: {py.name}")
            continue

        ctx = FeatureContext(
            pv_dir=pv_dir,
            full_dir=full_dir,
            out_root=out_root,
            tz=args.tz,
            atomic_write=True,
            parquet_compression="snappy",
        )
        out_dir = feature.out_dir(ctx)
        dest = out_dir / args.single_file_name

        if dest.exists() and not args.overwrite:
            print(f"[runner_onefile] 已存在单文件，跳过 {feature.name}: {dest}")
            continue

        print(f"=== 运行特征: {feature.name} | 文件: {py.name} | 日期数: {len(dates)} ===")
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        schema = pa.schema([
            pa.field("time", pa.timestamp("ns")),
            pa.field("symbol", pa.string()),
            pa.field("value", pa.float64()),
        ])

        writer = pq.ParquetWriter(tmp.as_posix(), schema=schema, compression=ctx.parquet_compression or "snappy")
        ok = skip = invalid = err = 0
        try:
            for i, d in enumerate(dates, 1):
                try:
                    df = feature.process_date(ctx, d)
                    if df is None:
                        skip += 1
                        continue
                    if "symbol" not in df.columns or "value" not in df.columns:
                        invalid += 1
                        print(f"[invalid] {feature.name} {d}: 缺少 ['symbol','value']")
                        continue
                    out = _normalize_for_write(df, d)
                    tbl = pa.Table.from_pandas(out, preserve_index=False)
                    writer.write_table(tbl)
                    ok += 1
                except Exception as exc:  # noqa: BLE001
                    err += 1
                    print(f"[error] {feature.name} {d}: {type(exc).__name__}: {exc}")

                if i % 50 == 0 or i == len(dates):
                    print(
                        f"[progress] {feature.name} {i}/{len(dates)} | ok={ok} skip={skip} invalid={invalid} err={err}",
                    )
        finally:
            writer.close()

        os.replace(tmp, dest)
        print(
            f"=== 完成: {feature.name} | 单文件输出 → {dest} | ok={ok} skip={skip} invalid={invalid} err={err} ===\n",
        )


if __name__ == "__main__":
    main()
