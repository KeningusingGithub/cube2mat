# runner.py
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import datetime as dt
import traceback
from typing import Optional, Iterable

from feature_base import BaseFeature, FeatureContext, get_dates_from_pv

# ---------- 动态加载 feature 模块 ----------
def load_feature_from_file(pyfile: Path) -> Optional[BaseFeature]:
    """
    支持两种写法：
    1) 在模块内定义变量 `feature = YourFeatureClass()`
    2) 在模块内定义某个 BaseFeature 的子类；若只有一个子类则用它
    """
    try:
        spec = spec_from_file_location(pyfile.stem, pyfile)
        if spec is None or spec.loader is None:
            print(f"[load] cannot load spec: {pyfile}")
            return None
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore

        # 1) 优先找变量 `feature`
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
            print(f"[load] {pyfile} 有多个 BaseFeature 子类，请显式提供 `feature = Class()`")
        else:
            print(f"[load] {pyfile} 未发现特征类或 `feature` 变量")
        return None
    except Exception:
        print(f"[load] error while importing {pyfile}:\n{traceback.format_exc()}")
        return None

def discover_features(feat_dir: Path, include: Optional[Iterable[str]] = None, exclude: Optional[Iterable[str]] = None) -> list[BaseFeature]:
    feats = []
    include = set(include or [])
    exclude = set(exclude or [])
    for py in sorted(feat_dir.glob("*.py")):
        if py.name.startswith("_"):        # 跳过模板/私有
            continue
        if include and py.stem not in include:
            continue
        if py.stem in exclude:
            continue
        feat = load_feature_from_file(py)
        if feat is not None:
            feats.append(feat)
    return feats

# ---------- 单日执行（供多进程调用） ----------
def run_one_date_for_feature(args):
    feat_file, ctx_dict, date_iso = args
    # 每个子进程独立 import，避免跨进程序列化复杂对象
    feature = load_feature_from_file(Path(feat_file))
    if feature is None:
        return (feat_file, date_iso, "load_failed")

    ctx = FeatureContext(**ctx_dict)
    date = dt.date.fromisoformat(date_iso)

    try:
        # 已有文件则跳过
        if feature.skip_if_exists and feature.already_done(ctx, date):
            return (feature.name, date_iso, "exists")

        df = feature.process_date(ctx, date)
        if df is None:
            return (feature.name, date_iso, "skip")
        if "symbol" not in df.columns or "value" not in df.columns:
            return (feature.name, date_iso, "invalid_output")

        feature.save_output(ctx, date, df)
        return (feature.name, date_iso, "ok")
    except Exception:
        return (feature.name, date_iso, f"error:{traceback.format_exc()}")

# ---------- 主逻辑 ----------
def parse_args():
    base_dir = Path(__file__).resolve().parent
    dataraw_root = base_dir.parent.parent.parent / "dataraw"
    ap = argparse.ArgumentParser(description="Feature runner: 扫描 features/*.py 并按日期批量计算")
    ap.add_argument("--pv-dir", default=str(dataraw_root / "us" / "basedata" / "close"))
    ap.add_argument("--full-dir", default=str(dataraw_root / "us" / "cubefull"))
    ap.add_argument("--out-root", default=str(dataraw_root / "us" / "cube2mat"))
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--start", default="2018-07-01", help="开始日期 YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD，默认= cubefull 目录最新日期")
    ap.add_argument("--features-dir", default=str(base_dir / "features"))
    ap.add_argument("--only", nargs="*", help="只运行这些特征（文件名不带 .py）")
    ap.add_argument("--exclude", nargs="*", help="排除这些特征")
    ap.add_argument("--processes", type=int, default=6)
    ap.add_argument("--chunksize", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true", help="忽略 skip_if_exists，强制重算")
    return ap.parse_args()

def main():
    args = parse_args()
    pv_dir = Path(args.pv_dir)
    full_dir = Path(args.full_dir)
    out_root = Path(args.out_root)
    feat_dir = Path(args.features_dir)

    # 结束日期默认取 cubefull 最新
    if args.end is None:
        latest = max((p.stem for p in full_dir.glob("*.parquet")), default=None)
        if latest is None:
            raise SystemExit("cubefull 目录为空，无法推断 end 日期")
        end_date = dt.datetime.strptime(latest, "%Y%m%d").date()
    else:
        end_date = dt.date.fromisoformat(args.end)

    start_date = dt.date.fromisoformat(args.start)

    # 扫描日期（根据 pv 目录）
    dates = get_dates_from_pv(pv_dir, start_date, end_date)
    if not dates:
        raise SystemExit("未发现任何有效日期（检查 pv_dir / 起止日期）")

    # 扫描特征
    # 注意：为便于多进程导入，这里传递“文件路径”，子进程各自 import
    feat_files = []
    for py in sorted(feat_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        if args.only and py.stem not in set(args.only):
            continue
        if args.exclude and py.stem in set(args.exclude):
            continue
        feat_files.append(py)

    if not feat_files:
        raise SystemExit("features 目录下没有可运行的 .py 特征文件")

    # 逐特征运行；每个特征内部对日期做并行
    for py in feat_files:
        # 先加载一次，拿到名字与输出目录（便于打印/创建）
        feature = load_feature_from_file(py)
        if feature is None:
            print(f"[runner] 跳过（加载失败）: {py.name}")
            continue

        # out_dir 准备
        ctx = FeatureContext(
            pv_dir=pv_dir,
            full_dir=full_dir,
            out_root=out_root,
            tz=args.tz,
            atomic_write=True,
            parquet_compression="snappy"
        )
        feature.out_dir(ctx)  # ensure exists

        # 如果没有 overwrite，而特征声明 skip_if_exists，则仍然允许你用 --overwrite 覆盖
        if args.overwrite:
            feature.skip_if_exists = False

        print(f"=== 运行特征: {feature.name} | 文件: {py.name} | 日期数: {len(dates)} ===")

        # 多进程映射
        tasks = [
            (str(py), ctx.__dict__, d.isoformat())
            for d in dates
        ]
        ok = err = skip = exist = invalid = 0
        with ProcessPoolExecutor(max_workers=args.processes) as ex:
            futs = [ex.submit(run_one_date_for_feature, t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                feat_name, dstr, status = f.result()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                elif status == "exists":
                    exist += 1
                elif status == "invalid_output":
                    invalid += 1
                    print(f"[invalid] {feat_name} {dstr}: 结果缺少 ['symbol','value']")
                else:
                    err += 1
                    print(f"[error] {feat_name} {dstr}: {status}")
                if i % 50 == 0 or i == len(futs):
                    print(f"[progress] {feature.name} {i}/{len(futs)} | ok={ok} exist={exist} skip={skip} invalid={invalid} err={err}")

        print(f"=== 完成: {feature.name} | ok={ok} exist={exist} skip={skip} invalid={invalid} err={err} ===\n")

if __name__ == "__main__":
    main()
