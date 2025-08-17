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


# ---------- 单特征单日冒烟测试（供多进程调用） ----------
def smoke_one_feature(args: Tuple[str, Dict[str, Any], str]) -> Tuple[str, str, str, str]:
    """
    返回: (pyfile, feature_name(or stem), status, info)
    status: PASS / FAIL / LOAD_FAIL
    """
    feat_file, base_ctx_dict, date_iso = args
    py_path = Path(feat_file)

    # 在子进程里各自 import，互不影响
    feature = load_feature_from_file(py_path)
    if feature is None:
        return (py_path.name, py_path.stem, "LOAD_FAIL", "加载/导入失败")

    # 防止被“已存在”逻辑跳过
    try:
        feature.skip_if_exists = False
    except Exception:
        pass

    # 强行禁用保存：猴子补丁 save_output
    try:
        feature.save_output = lambda *a, **k: None  # type: ignore[attr-defined]
    except Exception:
        pass

    # 为本次子任务准备临时 out_root，退出即删除
    date = dt.date.fromisoformat(date_iso)
    try:
        with tempfile.TemporaryDirectory(prefix=f"feat_smoke_{py_path.stem}_") as tmpdir:
            ctx_dict = dict(base_ctx_dict)
            ctx_dict["out_root"] = Path(tmpdir)
            # 对于极个别需要 out_dir 存在的实现，确保目录创建由 FeatureContext/feature 自行处理
            ctx = FeatureContext(**ctx_dict)

            # 真正计算（只跑一天）
            df = feature.process_date(ctx, date)

            # ---- 校验 ----
            if df is None:
                return (py_path.name, getattr(feature, "name", py_path.stem), "FAIL", "process_date 返回 None/被跳过")

            # DataFrame 基本格式
            if not hasattr(df, "columns"):
                return (py_path.name, getattr(feature, "name", py_path.stem), "FAIL", "输出不是 DataFrame")

            missing_cols = [c for c in ("symbol", "value") if c not in df.columns]
            if missing_cols:
                return (
                    py_path.name,
                    getattr(feature, "name", py_path.stem),
                    "FAIL",
                    f"缺少必要列: {missing_cols}",
                )

            if df.empty:
                return (py_path.name, getattr(feature, "name", py_path.stem), "FAIL", "输出为空 DataFrame")

            # 质量：value 至少有一个非空
            non_null_mask = df["value"].notna()
            if not bool(non_null_mask.any()):
                return (
                    py_path.name,
                    getattr(feature, "name", py_path.stem),
                    "FAIL",
                    "value 全为 NaN/None",
                )

            # symbol 也不应全空
            if "symbol" in df.columns and bool(getattr(df["symbol"], "isna", lambda: False)().all()):
                return (
                    py_path.name,
                    getattr(feature, "name", py_path.stem),
                    "FAIL",
                    "symbol 全为空",
                )

            # 通过
            info = f"rows={len(df)}, non_null_value={int(non_null_mask.sum())}"
            return (py_path.name, getattr(feature, "name", py_path.stem), "PASS", info)

    except Exception:
        return (py_path.name, getattr(feature, "name", py_path.stem), "FAIL", f"异常: {traceback.format_exc()}")


# ---------- 主逻辑 ----------
def parse_args():
    base_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Feature smoke tester: 扫描 features/*.py，单日跑通检查（不落盘）"
    )
    ap.add_argument("--pv-dir", default="/home/ubuntu/dataraw/us/pv")
    ap.add_argument("--full-dir", default="/home/ubuntu/dataraw/us/cubefull")
    ap.add_argument("--features-dir", default=str(base_dir / "features"))

    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--date", default=None, help="测试日期 YYYY-MM-DD；缺省时自动选择最新可跑日期")
    ap.add_argument("--start", default="2018-07-01", help="候选日期范围起点（自动选日期时生效）")
    ap.add_argument("--end", default=None, help="候选日期范围终点（自动选日期时生效；缺省取 cubefull 最新文件日）")

    ap.add_argument("--only", nargs="*", help="只测试这些特征（文件名不带 .py）")
    ap.add_argument("--exclude", nargs="*", help="排除这些特征")
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--chunksize", type=int, default=1)  # 单日，无需大 chunks

    return ap.parse_args()


def pick_test_date(
    pv_dir: Path,
    full_dir: Path,
    start: dt.date,
    end: Optional[dt.date],
    explicit_date: Optional[str],
) -> dt.date:
    if explicit_date:
        return dt.date.fromisoformat(explicit_date)

    # 结束日期默认取 cubefull 最新
    if end is None:
        latest = max((p.stem for p in full_dir.glob("*.parquet")), default=None)
        if latest is None:
            raise SystemExit("cubefull 目录为空，无法推断 end 日期；请用 --date 指定具体测试日")
        end = dt.datetime.strptime(latest, "%Y%m%d").date()

    # 从 pv 目录挑可跑日期，取最新一个
    dates = get_dates_from_pv(pv_dir, start, end)
    if not dates:
        raise SystemExit("未发现任何有效测试日期（检查 pv_dir / 起止日期）")
    return dates[-1]


def main():
    args = parse_args()

    pv_dir = Path(args.pv_dir)
    full_dir = Path(args.full_dir)
    feat_dir = Path(args.features_dir)

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end) if args.end else None

    test_date = pick_test_date(pv_dir, full_dir, start_date, end_date, args.date)
    print(f"[*] 测试日期: {test_date.isoformat()} (仅单日)")

    feat_files = discover_feature_files(feat_dir, args.only, args.exclude)
    if not feat_files:
        raise SystemExit("features 目录下没有可运行的 .py 特征文件")

    print(f"[*] 待测特征数量: {len(feat_files)}")
    base_ctx_dict = dict(
        pv_dir=pv_dir,
        full_dir=full_dir,
        out_root=Path(tempfile.gettempdir()) / "feat_smoke_dummy",  # 会被子进程覆盖为临时目录
        tz=args.tz,
        atomic_write=False,  # 防御写入
        parquet_compression="snappy",
    )

    tasks = [(str(py), base_ctx_dict, test_date.isoformat()) for py in feat_files]

    # 跑起来
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

    # 约定：若存在失败，返回非零码，便于 CI 集成
    import sys
    sys.exit(0 if fail == 0 and load_fail == 0 else 1)


if __name__ == "__main__":
    main()
