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
LOG_PATH = Path(__file__).with_name("quote_smoke_fail.log")


# ---------- 动态加载 feature 模块（与原 runner 等价） ----------
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

        # 1) 优先找变量 feature
        if hasattr(mod, "feature"):
            feat = getattr(mod, "feature")
            if isinstance(feat, BaseFeature):
                return feat
            else:
                print(f"[load] {pyfile} 'feature' 不是 BaseFeature 实例")
                return None

        # 2) 兜底：找唯一的 BaseFeature 子类
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


# ---------- feature 文件扫描 ----------
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


# ---------- 日期扫描 ----------
def get_dates_from_quote_root(quote_root: Path, start: dt.date, end: dt.date) -> list[dt.date]:
    dates: list[dt.date] = []
    for p in quote_root.iterdir():
        if p.is_dir() and len(p.name) == 8 and p.name.isdigit():
            try:
                d = dt.datetime.strptime(p.name, "%Y%m%d").date()
            except ValueError:
                continue
            if start <= d <= end:
                dates.append(d)
    return sorted(dates)


def pick_test_date(
    pv_dir: Path,
    quote_root: Path,
    start: dt.date,
    end: Optional[dt.date],
    explicit_date: Optional[str],
) -> dt.date:
    if explicit_date:
        return dt.date.fromisoformat(explicit_date)

    if end is None:
        # 扫描宽区间，取 pv 与 quote 的交集最新日期
        pv_dates = set(get_dates_from_pv(pv_dir, start, dt.date(2100, 1, 1)))
        quote_dates = set(get_dates_from_quote_root(quote_root, start, dt.date(2100, 1, 1)))
    else:
        pv_dates = set(get_dates_from_pv(pv_dir, start, end))
        quote_dates = set(get_dates_from_quote_root(quote_root, start, end))

    candidates = sorted(pv_dates & quote_dates)
    if not candidates:
        raise SystemExit("未发现任何有效测试日期（检查 pv_dir / quote_root）")
    return candidates[-1]


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
        with tempfile.TemporaryDirectory(prefix=f"quote_feat_smoke_{py_path.stem}_") as tmpdir:
            ctx_dict = dict(base_ctx_dict)
            ctx_dict["out_root"] = Path(tmpdir)
            ctx = FeatureContext(**ctx_dict)
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
        description="Quote Feature smoke tester: 扫描 quote_features/*.py，单日跑通检查（不落盘）"
    )
    ap.add_argument("--pv-dir", default=str(dataraw_root / "us" / "basedata" / "close"))
    ap.add_argument("--full-dir", default=str(dataraw_root / "us" / "cubefull"))
    ap.add_argument("--quote-root", default=str(dataraw_root / "us" / "quote_onefile"))
    ap.add_argument("--features-dir", default=str(base_dir / "quote_features"))

    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--date", default=None, help="测试日期 YYYY-MM-DD；缺省时自动选择最新可跑日期")
    ap.add_argument("--start", default="2018-07-01", help="候选日期范围起点（自动选日期时生效）")
    ap.add_argument("--end", default=None, help="候选日期范围终点（自动选日期时生效）")

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
    quote_root = Path(args.quote_root)
    feat_dir = Path(args.features_dir)

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end) if args.end else None

    test_date = pick_test_date(pv_dir, quote_root, start_date, end_date, args.date)
    print(f"[*] 测试日期: {test_date.isoformat()} (仅单日)")

    feat_files = discover_feature_files(feat_dir, args.only, args.exclude)
    if not feat_files:
        raise SystemExit("quote_features 目录下没有可运行的 .py 特征文件")

    print(f"[*] 待测特征数量: {len(feat_files)}")
    base_ctx_dict = dict(
        pv_dir=pv_dir,
        full_dir=full_dir,
        quote_root=str(quote_root),
        out_root=Path(tempfile.gettempdir()) / "quote_feat_smoke_dummy",  # 将被子进程覆盖
        tz=args.tz,
        atomic_write=False,
        parquet_compression="snappy",
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
