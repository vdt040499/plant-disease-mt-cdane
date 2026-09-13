"""Train one arm.

    python scripts/train.py --arm cdane
    python scripts/train.py --arm mt

Both arms share a seed, a batch order and a test split, so their accuracies are
directly comparable. Results are written to <root>/results/<arm>_s<seed>.json
and skipped if that file already exists; delete it to re-run. Checkpoints are
written every `--log-every` epochs and resumed automatically.
"""

import argparse
import json

import _bootstrap  # noqa: F401
from plantuda.config import DataConfig, MeanTeacherConfig, Paths, TrainConfig
from plantuda.data import build_loaders, load_cached_fbr_samples, load_ppd_samples
from plantuda.engine import train_arm
from plantuda.seeding import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["cdane", "mt"], required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument("--min-select", type=int, default=TrainConfig.min_select_epoch)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--cons-start", type=int, default=MeanTeacherConfig.cons_start)
    parser.add_argument("--cons-ramp-len", type=int, default=MeanTeacherConfig.cons_ramp_len)
    parser.add_argument("--cons-max", type=float, default=MeanTeacherConfig.cons_max)
    parser.add_argument("--force", action="store_true", help="ignore a cached result")
    args = parser.parse_args()

    use_mt = args.arm == "mt"
    paths = (Paths(args.root) if args.root else Paths.from_env()).ensure()
    result_path = paths.results / f"{args.arm}_s{args.seed}.json"

    if result_path.exists() and not args.force:
        print(f"[skip] result already exists: {result_path}")
        print(json.dumps(json.loads(result_path.read_text()), indent=2))
        return

    data_cfg = DataConfig(seed=args.seed)
    train_cfg = TrainConfig(
        seed=args.seed,
        max_epochs=args.epochs,
        min_select_epoch=args.min_select,
        log_every=args.log_every,
    )
    mt_cfg = MeanTeacherConfig(
        cons_max=args.cons_max,
        cons_start=args.cons_start,
        cons_ramp_len=args.cons_ramp_len,
    )

    fbr_samples = load_cached_fbr_samples(paths.fbr_cache)
    ppd_samples = load_ppd_samples(paths.plant_pathology)
    loaders = build_loaders(fbr_samples, ppd_samples, data_cfg)

    print("=" * 60)
    print(f"ARM: {args.arm}  seed={args.seed}  epochs={args.epochs}")
    print(
        f"source train={len(loaders.source_train.dataset)} "
        f"val={len(loaders.source_val.dataset)} "
        f"target={len(loaders.target_three_view.dataset)} "
        f"test={len(loaders.test.dataset)}"
    )
    print("=" * 60)

    result = train_arm(
        loaders,
        use_mt=use_mt,
        device=resolve_device(),
        train_cfg=train_cfg,
        mt_cfg=mt_cfg,
        ckpt_dir=paths.checkpoints,
    )

    result_path.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"\n{result.arm}: test accuracy {result.acc_selected:.2f}%")
    print(f"written: {result_path}")


if __name__ == "__main__":
    main()
