# quote_runner.py
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import datetime as dt
import traceback
from typing import Optional, Iterable, List, Set, Dict, Any

from feature_base import BaseFeature, FeatureContext, get_dates_from_pv

# ---------- 动态加载 feature 模块（与原 runner 相同逻辑） ----------
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


# ---------- 日期扫描（基于 quote 根目录） ----------
def infer_latest_date_from_quote_root(quote_root: Path) -> Optional[dt.date]:
    """
    扫描 quote_root 下的形如 YYYYMMDD 的子目录，取其中最新的日期。
    """
    latest: Optional[dt.date] = None
    for p in quote_root.iterdir():
        if p.is_dir() and len(p.name) == 8 and p.name.isdigit():
            try:
                d = dt.datetime.strptime(p.name, "%Y%m%d").date()
            except ValueError:
                continue
            if latest is None or d > latest:
                latest = d
    return latest


def get_dates_from_quote_root(quote_root: Path, start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    """
    返回位于 [start_date, end_date] 且在 quote_root 下存在对应 YYYYMMDD 目录的日期列表。
    """
    avail: Set[dt.date] = set()
    for p in quote_root.iterdir():
        if p.is_dir() and len(p.name) == 8 and p.name.isdigit():
            try:
                d = dt.datetime.strptime(p.name, "%Y%m%d").date()
            except ValueError:
                continue
            if start_date <= d <= end_date:
                avail.add(d)
    return sorted(avail)


# ---------- 单日执行（供多进程调用） ----------
def run_one_date_for_feature(args):
    feat_file, ctx_kwargs, extra_ctx_attrs, date_iso = args
    # 每个子进程独立 import，避免跨进程序列化复杂对象
    feature = load_feature_from_file(Path(feat_file))
    if feature is None:
        return (feat_file, date_iso, "load_failed")

    # 仅用构造函数支持的参数初始化
    ctx = FeatureContext(**ctx_kwargs)
    # 再设置额外上下文字段（例如 quote_root）
    for k, v in (extra_ctx_attrs or {}).items():
        setattr(ctx, k, v)

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
    ap = argparse.ArgumentParser(description="Quote Feature runner: 扫描 quote_features/*.py 并按日期批量计算")
    ap.add_argument("--pv-dir", default="/home/ubuntu/dataraw/us/pv", help="用于样本与日期扫描（若 --date-source=pv 或 intersect）")
    ap.add_argument("--full-dir", default="/home/ubuntu/dataraw/us/cubefull", help="与 FeatureContext 兼容所需（即使 quote 特征可能用不上）")
    ap.add_argument("--quote-root", default="/home/ubuntu/dataraw/us/quote", help="quote 数据根目录（YYYYMMDD/SYMBOL.parquet）")
    ap.add_argument("--out-root", default="/home/ubuntu/dataraw/us/quote_features", help="特征输出根目录")
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--start", default="2018-07-01", help="开始日期 YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD；默认按 --date-source 推断")
    ap.add_argument("--features-dir", default=str(base_dir / "quote_features"), help="特征脚本所在目录")
    ap.add_argument("--only", nargs="*", help="只运行这些特征（文件名不带 .py）")
    ap.add_argument("--exclude", nargs="*", help="排除这些特征")
    ap.add_argument("--processes", type=int, default=6)
    ap.add_argument("--chunksize", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true", help="忽略 skip_if_exists，强制重算")
    ap.add_argument("--date-source", choices=["pv", "quote", "intersect"], default="pv",
                    help="日期来源：pv=从 pv 目录（支持 YYYY_M_D.parquet）；quote=从 quote 根目录；intersect=两者交集")
    return ap.parse_args()


def main():
    args = parse_args()
    pv_dir = Path(args.pv_dir)
    full_dir = Path(args.full_dir)
    quote_root = Path(args.quote_root)
    out_root = Path(args.out_root)
    feat_dir = Path(args.features_dir)

    # 结束日期默认推断
    if args.end is None:
        if args.date_source in ("pv", "intersect"):
            # 用超宽区间抓全量 pv 日期，取最大值
            all_pv_dates = get_dates_from_pv(pv_dir, dt.date(1900, 1, 1), dt.date(2100, 1, 1))
            if not all_pv_dates:
                raise SystemExit("pv 目录为空或文件名无法解析为日期，无法推断 end 日期（可显式指定 --end 或改用 --date-source=quote）")
            end_date = max(all_pv_dates)
        else:  # quote
            latest = infer_latest_date_from_quote_root(quote_root)
            if latest is None:
                raise SystemExit("quote_root 下未发现 YYYYMMDD 子目录，无法推断 end 日期（请确认路径或显式指定 --end）")
            end_date = latest
    else:
        end_date = dt.date.fromisoformat(args.end)

    start_date = dt.date.fromisoformat(args.start)

    # 扫描日期
    if args.date_source == "pv":
        dates = get_dates_from_pv(pv_dir, start_date, end_date)
    elif args.date_source == "quote":
        dates = get_dates_from_quote_root(quote_root, start_date, end_date)
    else:  # intersect
        d1 = set(get_dates_from_pv(pv_dir, start_date, end_date))
        d2 = set(get_dates_from_quote_root(quote_root, start_date, end_date))
        dates = sorted(d1 & d2)

    if not dates:
        raise SystemExit("未发现任何有效日期（检查 pv_dir/quote_root 与起止日期、以及 --date-source）")

    # 扫描特征（传“文件路径”，子进程各自 import）
    feat_files: List[Path] = []
    for py in sorted(feat_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        if args.only and py.stem not in set(args.only):
            continue
        if args.exclude and py.stem in set(args.exclude):
            continue
        feat_files.append(py)

    if not feat_files:
        raise SystemExit("quote_features 目录下没有可运行的 .py 特征文件")

    # 逐特征运行；每个特征内部对日期做并行
    for py in feat_files:
        # 先加载一次，拿到名字与输出目录（便于打印/创建）
        feature = load_feature_from_file(py)
        if feature is None:
            print(f"[runner] 跳过（加载失败）: {py.name}")
            continue

        # 仅用构造函数支持的字段初始化 ctx
        ctx = FeatureContext(
            pv_dir=pv_dir,
            full_dir=full_dir,
            out_root=out_root,
            tz=args.tz,
            atomic_write=True,
            parquet_compression="snappy"
        )
        # 再挂载额外的上下文字段
        setattr(ctx, "quote_root", str(quote_root))

        feature.out_dir(ctx)  # ensure exists

        if args.overwrite:
            feature.skip_if_exists = False

        print(f"=== 运行特征: {feature.name} | 文件: {py.name} | 日期数: {len(dates)} | date_source={args.date_source} ===")

        # 子进程任务参数：把构造参数与“额外上下文”分开传
        ctx_kwargs: Dict[str, Any] = {
            "pv_dir": pv_dir,
            "full_dir": full_dir,
            "out_root": out_root,
            "tz": args.tz,
            "atomic_write": True,
            "parquet_compression": "snappy",
        }
        extra_ctx_attrs: Dict[str, Any] = {
            "quote_root": str(quote_root),
        }

        tasks = [(str(py), ctx_kwargs, extra_ctx_attrs, d.isoformat()) for d in dates]
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
