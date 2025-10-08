from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import tempfile
import traceback
from typing import Optional, Iterable, Tuple, Dict, Any

from feature_base import BaseFeature, FeatureContext, get_dates_from_pv

# 单一日志文件，位于本脚本同路径
LOG_PATH = Path(__file__).with_name("trades_smoke_fail.log")


# ---------- 动态加载 feature 模块（与 quote_smoke_test 等价） ----------
def load_feature_from_file(pyfile: Path) -> Optional[BaseFeature]:
    """支持两种写法：
    1) 在模块内定义变量 feature = YourFeatureClass()
    2) 在模块内定义某个 BaseFeature 的子类；若只有一个子类则用它
    """
    try:
        spec = spec_from_file_location(pyfile.stem, pyfile)
        if spec is None or spec.loader is None:
            print(f"[load] cannot load spec: {pyfile}")
            return None
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore

        if hasattr(mod, "feature"):
            feat = getattr(mod, "feature")
            if isinstance(feat, BaseFeature):
                return feat
            else:
                print(f"[load] {pyfile} 'feature' 不是 BaseFeature 实例")
                return None

        candidates = []
        for v in mod.__dict__.values():
            if isinstance(v, type) and issubclass(v, BaseFeature) and v is not BaseFeature:
                candidates.append(v)
        if len(candidates) == 1:
            return candidates[0]()
        if len(candidates) > 1:
            print(f"[load] {pyfile} 有多个 BaseFeature 子类，请显式提供 feature = Class()")
        else:
            print(f"[load] {pyfile} 未发现特征类或 feature 变量")
        return None
    except Exception:
        print(f"[load] error while importing {pyfile}:\n{traceback.format_exc()}")
        return None


# ---------- trade 布局探测 & 日期扫描（兼容 dir 与 onefile） ----------
def detect_trade_layout(trade_root: Path) -> str:
    """返回 'onefile' | 'dir' | 'unknown'"""
    has_onefile = any(
        p.is_file() and p.suffix.lower() in (".parquet", ".parq") and len(p.stem) == 8 and p.stem.isdigit()
        for p in trade_root.iterdir()
    )
    has_dir = any(
        p.is_dir() and len(p.name) == 8 and p.name.isdigit()
        for p in trade_root.iterdir()
    )
    if has_onefile and not has_dir:
        return "onefile"
    if has_dir and not has_onefile:
        return "dir"
    if has_onefile and has_dir:
        print("[detect] 同时检测到 onefile 与 dir，默认使用 onefile（可用 --trade-layout 指定）")
        return "onefile"
    return "unknown"


def get_dates_from_trade(trade_root: Path, start: dt.date, end: dt.date, layout: str) -> list[dt.date]:
    dates: set[dt.date] = set()
    if layout == "dir":
        for p in trade_root.iterdir():
            if p.is_dir() and len(p.name) == 8 and p.name.isdigit():
                try:
                    d = dt.datetime.strptime(p.name, "%Y%m%d").date()
                except ValueError:
                    continue
                if start <= d <= end:
                    dates.add(d)
    elif layout == "onefile":
        for p in trade_root.iterdir():
            if p.is_file() and p.suffix.lower() in (".parquet", ".parq") and len(p.stem) == 8 and p.stem.isdigit():
                try:
                    d = dt.datetime.strptime(p.stem, "%Y%m%d").date()
                except ValueError:
                    continue
                if start <= d <= end:
                    dates.add(d)
    else:
        # layout 异常时直接返回空列表，由调用方处理
        pass
    return sorted(dates)


def discover_feature_files(
    feat_dir: Path,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> list[Path]:
    include = set(include or [])
    exclude = set(exclude or [])
    files: list[Path] = []
    for py in sorted(feat_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        if include and py.stem not in include:
            continue
        if py.stem in exclude:
            continue
        files.append(py)
    return files


def pick_test_date(
    pv_dir: Path,
    trade_root: Path,
    start: dt.date,
    end: Optional[dt.date],
    explicit_date: Optional[str],
    date_source: str,
    layout: str,
) -> dt.date:
    if explicit_date:
        return dt.date.fromisoformat(explicit_date)

    scan_end = end or dt.date(2100, 1, 1)

    if date_source == "pv":
        candidates = get_dates_from_pv(pv_dir, start, scan_end)
    elif date_source == "trade":
        candidates = get_dates_from_trade(trade_root, start, scan_end, layout)
    else:  # intersect
        pv_dates = set(get_dates_from_pv(pv_dir, start, scan_end))
        trade_dates = set(get_dates_from_trade(trade_root, start, scan_end, layout))
        candidates = sorted(pv_dates & trade_dates)

    if not candidates:
        raise SystemExit("未发现任何有效测试日期（检查 pv_dir / trade_root）")
    return candidates[-1]


# ---------- 在 Context 上注入与 runner 等价的 trade helpers（与 quote 版思路相同） ----------
def _attach_trade_helpers(ctx: FeatureContext):
    import importlib
    trade_root = Path(getattr(ctx, "trade_root"))
    layout = getattr(ctx, "trade_layout", "auto")

    # 自动识别（仅用于 smoke；runner 中由 CLI 指定或探测）
    if layout == "auto":
        has_onefile = any(p.is_file() and p.suffix.lower() in (".parquet", ".parq") and len(p.stem) == 8 and p.stem.isdigit()
                          for p in trade_root.iterdir())
        layout = "onefile" if has_onefile else "dir"

    if layout == "onefile":
        ds = importlib.import_module("pyarrow.dataset")
        pq = importlib.import_module("pyarrow.parquet")

        def get_day_parquet(date: dt.date) -> Path:
            return trade_root / f"{date.strftime('%Y%m%d')}.parquet"

        def iterate_day_for_symbol(date: dt.date, symbol: str,
                                   columns: list[str] | None = None, batch_size: int = 200_000):
            path = get_day_parquet(date)
            dataset = ds.dataset(str(path), format="parquet")
            predicate = (ds.field("symbol") == symbol)
            for batch in dataset.to_batches(filter=predicate, columns=columns, batch_size=batch_size):
                yield batch.to_pandas(copy=False)

        def iterate_day_in_batches(date: dt.date,
                                   columns: list[str] | None = None, batch_size: int = 200_000):
            path = get_day_parquet(date)
            pfq = pq.ParquetFile(str(path))
            for batch in pfq.iter_batches(batch_size=batch_size, columns=columns):
                yield batch.to_pandas(copy=False)

        setattr(ctx, "get_day_parquet", get_day_parquet)
        setattr(ctx, "iterate_day_for_symbol", iterate_day_for_symbol)
        setattr(ctx, "iterate_day_in_batches", iterate_day_in_batches)
    else:
        import pandas as pd

        def get_day_dir(date: dt.date) -> Path:
            return trade_root / date.strftime("%Y%m%d")

        def get_symbol_parquet(date: dt.date, symbol: str) -> Path:
            return get_day_dir(date) / f"{symbol}.parquet"

        def iterate_day_for_symbol(date: dt.date, symbol: str,
                                   columns: list[str] | None = None, batch_size: int = 200_000):
            path = get_symbol_parquet(date, symbol)
            if not path.exists():
                return
            df = pd.read_parquet(path, columns=columns)
            yield df

        def iterate_day_in_batches(date: dt.date,
                                   columns: list[str] | None = None, batch_size: int = 200_000):
            day_dir = get_day_dir(date)
            if not day_dir.exists():
                return
            for f in sorted(day_dir.glob("*.parquet")):
                try:
                    df = pd.read_parquet(f, columns=columns)
                except Exception:
                    continue
                yield df

        setattr(ctx, "get_day_dir", get_day_dir)
        setattr(ctx, "get_symbol_parquet", get_symbol_parquet)
        setattr(ctx, "iterate_day_for_symbol", iterate_day_for_symbol)
        setattr(ctx, "iterate_day_in_batches", iterate_day_in_batches)


# ---------- 单特征单日冒烟测试（供多进程调用） ----------
def smoke_one_feature(args: Tuple[str, Dict[str, Any], str]) -> Tuple[str, str, str, str]:
    """
    返回: (pyfile, feature_name(or stem), status, info)
    status: PASS / FAIL / LOAD_FAIL
    """
    feat_file, base_ctx_dict, date_iso = args
    py_path = Path(feat_file)

    feature = load_feature_from_file(py_path)

    def _make_result(status: str, info: str, data: str | None = None) -> Tuple[str, str, str, str]:
        feat_name = getattr(feature, "name", py_path.stem)
        if status != "PASS":
            try:
                with open(LOG_PATH, "a", encoding="utf-8") as fh:
                    fh.write(f"{py_path.name} | {feat_name} | {status} | {info}\n")
                    if data:
                        fh.write(f"{data}\n")
                    fh.write("\n")
            except Exception:
                pass
        return (py_path.name, feat_name, status, info)

    if feature is None:
        return _make_result("LOAD_FAIL", "加载/导入失败")

    try:
        feature.skip_if_exists = False
    except Exception:
        pass

    try:
        feature.save_output = lambda *a, **k: None  # type: ignore[attr-defined]
    except Exception:
        pass

    date = dt.date.fromisoformat(date_iso)
    try:
        with tempfile.TemporaryDirectory(prefix=f"trade_feat_smoke_{py_path.stem}_") as tmpdir:
            ctx_dict = dict(base_ctx_dict)
            ctx_dict["out_root"] = Path(tmpdir)
            trade_root_val = ctx_dict.pop("trade_root", None)
            trade_layout_val = ctx_dict.pop("trade_layout", "auto")
            ctx = FeatureContext(**ctx_dict)
            if trade_root_val is not None:
                setattr(ctx, "trade_root", trade_root_val)
            setattr(ctx, "trade_layout", trade_layout_val)

            # 为最大兼容性，注入与 runner 等价的 helpers（特征若直接依赖 ctx.iterate_* 也能跑）
            try:
                _attach_trade_helpers(ctx)
            except Exception:
                pass

            df = feature.process_date(ctx, date)

            if df is None:
                return _make_result("FAIL", "process_date 返回 None/被跳过")
            if not hasattr(df, "columns"):
                return _make_result("FAIL", "输出不是 DataFrame", data=str(df))
            missing_cols = [c for c in ("symbol", "value") if c not in df.columns]
            if missing_cols:
                return _make_result(
                    "FAIL", f"缺少必要列: {missing_cols}", data=df.head().to_csv(index=False)
                )
            if df.empty:
                return _make_result(
                    "FAIL", "输出为空 DataFrame", data=df.head().to_csv(index=False)
                )
            non_null_mask = df["value"].notna()
            if not bool(non_null_mask.any()):
                return _make_result(
                    "FAIL", "value 全为 NaN/None", data=df.head().to_csv(index=False)
                )
            if "symbol" in df.columns and bool(getattr(df["symbol"], "isna", lambda: False)().all()):
                return _make_result(
                    "FAIL", "symbol 全为空", data=df.head().to_csv(index=False)
                )
            info = f"rows={len(df)}, non_null_value={int(non_null_mask.sum())}"
            return _make_result("PASS", info)
    except Exception:
        return _make_result("FAIL", f"异常: {traceback.format_exc()}")


# ---------- 主逻辑 ----------
def parse_args():
    base_dir = Path(__file__).resolve().parent
    dataraw_root = base_dir.parent.parent.parent / "dataraw"
    ap = argparse.ArgumentParser(
        description="Trade Feature smoke tester: 扫描 trades_features/*.py，单日跑通检查（不落盘）"
    )
    ap.add_argument("--pv-dir", default=str(dataraw_root / "us" / "basedata" / "close"))
    ap.add_argument("--full-dir", default=str(dataraw_root / "us" / "cubefull"))
    ap.add_argument("--trade-root", default=str(dataraw_root / "us" / "trade_onefile"))
    ap.add_argument("--features-dir", default=str(base_dir / "trades_features"))

    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--date", default=None, help="测试日期 YYYY-MM-DD；缺省时自动选择最新可跑日期")
    ap.add_argument("--start", default="2018-07-01", help="候选日期范围起点（自动选日期时生效）")
    ap.add_argument("--end", default=None, help="候选日期范围终点（自动选日期时生效）")

    ap.add_argument(
        "--date-source",
        choices=["pv", "trade", "intersect"],
        default="intersect",
        help="测试日期来源：pv=仅 pv 目录；trade=仅 trade 目录；intersect=两者交集",
    )
    ap.add_argument(
        "--trade-layout",
        choices=["auto", "dir", "onefile"],
        default="auto",
        help="trade 数据布局：auto=自动探测；dir=YYYYMMDD/SYMBOL.parquet；onefile=YYYYMMDD.parquet",
    )

    ap.add_argument("--only", nargs="*", help="只测试这些特征（文件名不带 .py）")
    ap.add_argument("--exclude", nargs="*", help="排除这些特征")
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--chunksize", type=int, default=1)

    return ap.parse_args()


def main():
    args = parse_args()

    try:
        LOG_PATH.unlink()
    except FileNotFoundError:
        pass

    pv_dir = Path(args.pv_dir)
    full_dir = Path(args.full_dir)
    trade_root = Path(args.trade_root)
    feat_dir = Path(args.features_dir)

    if args.trade_layout == "auto":
        layout = detect_trade_layout(trade_root)
        if layout == "unknown":
            raise SystemExit(
                "trade_root 下既无 YYYYMMDD 目录也无 YYYYMMDD.parquet 文件，无法识别布局（可用 --trade-layout 指定）"
            )
    else:
        layout = args.trade_layout

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end) if args.end else None

    test_date = pick_test_date(
        pv_dir, trade_root, start_date, end_date, args.date, args.date_source, layout
    )
    print(f"[*] 测试日期: {test_date.isoformat()} (仅单日)")

    feat_files = discover_feature_files(feat_dir, args.only, args.exclude)

    if not feat_files:
        raise SystemExit("trades_features 目录下没有可运行的 .py 特征文件")

    print(f"[*] 待测特征数量: {len(feat_files)}")
    base_ctx_dict = dict(
        pv_dir=pv_dir,
        full_dir=full_dir,
        trade_root=str(trade_root),
        out_root=Path(tempfile.gettempdir()) / "trade_feat_smoke_dummy",  # 将被子进程覆盖
        tz=args.tz,
        atomic_write=False,
        parquet_compression="snappy",
        trade_layout=layout,
    )

    tasks = [(str(py), base_ctx_dict, test_date.isoformat()) for py in feat_files]

    results = []
    ok = fail = load_fail = 0
    with ProcessPoolExecutor(max_workers=args.processes) as ex:
        futs = [ex.submit(smoke_one_feature, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            pyname, feat_name, status, info = f.result()
            if status == "PASS":
                ok += 1
                print(f"[PASS] {feat_name:<30} | {pyname:<30} | {info}")
            elif status == "LOAD_FAIL":
                load_fail += 1
                print(f"[LOAD_FAIL] {feat_name:<30} | {pyname:<30} | {info}")
            else:
                fail += 1
                print(f"[FAIL] {feat_name:<30} | {pyname:<30} | {info}")
            results.append((pyname, feat_name, status, info))

    print("\n=== 冒烟测试完成 ===")
    print(f"总计: {len(results)} | 通过: {ok} | 加载失败: {load_fail} | 失败: {fail}")

    import sys
    sys.exit(0 if fail == 0 and load_fail == 0 else 1)


if __name__ == "__main__":
    main()
