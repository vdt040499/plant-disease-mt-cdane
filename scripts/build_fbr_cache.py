"""Pre-compute the FBR source images.

    python scripts/build_fbr_cache.py

Segments each PlantVillage leaf with SAM and composites it onto a patch of a
real field photograph. Backgrounds are drawn from the adaptation split only, so
no test-set imagery reaches training. Idempotent: existing cache entries are
reused, so an interrupted run can simply be restarted.
"""

import argparse

import _bootstrap  # noqa: F401
from plantuda.config import DataConfig, Paths
from plantuda.data import load_ppd_samples, load_pvd_samples, split_ppd, subset_samples
from plantuda.seeding import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=DataConfig.seed)
    args = parser.parse_args()

    # Imported here so --help works without OpenCV and segment-anything, which
    # only this one step needs.
    from plantuda.fbr import build_fbr_cache, load_sam

    paths = Paths(args.root) if args.root else Paths.from_env()
    paths.ensure()
    cfg = DataConfig(seed=args.seed)
    device = resolve_device()

    pvd = load_pvd_samples(
        paths.plant_village_images, max_per_class=cfg.max_per_class, seed=cfg.seed
    )
    ppd = load_ppd_samples(paths.plant_pathology)
    target_subset, _ = split_ppd(ppd, target_ratio=cfg.ppd_target_ratio, seed=cfg.seed)
    bg_paths = [p for p, _ in subset_samples(target_subset)]
    print(f"Background pool (adaptation split only): {len(bg_paths)} images")

    checkpoint = args.sam_checkpoint or str(paths.root / "sam_vit_b_01ec64.pth")
    predictor = load_sam(checkpoint, model_type="vit_b", device=device)

    build_fbr_cache(pvd, bg_paths, predictor, paths.fbr_cache, seed=cfg.seed)


if __name__ == "__main__":
    main()
