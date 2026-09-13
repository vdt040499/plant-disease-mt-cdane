"""Verify that both arms see the same source data stream.

    python scripts/rng_probe.py

The whole comparison rests on one claim: the Mean Teacher arm and the baseline
consume the same source images, in the same order, with the same augmentation.
That claim breaks silently -- one extra module construction is enough -- so it
is measured rather than assumed.

The probe replays each arm's RNG consumption (construction order, the forked
teacher, the per-epoch loader seed) and compares a signature of every source
batch. No backward pass, so it takes about a minute and runs on CPU.
"""

import argparse

import torch

import _bootstrap  # noqa: F401
from plantuda.config import DataConfig, NUM_CLASSES, Paths
from plantuda.data import build_loaders, load_cached_fbr_samples, load_ppd_samples
from plantuda.models import CDANDisc, Classifier, FeatureExtractor, GradRevLayer
from plantuda.models.mean_teacher import create_teacher
from plantuda.seeding import seed_loader_epoch, set_seed


def probe_source_stream(loaders, seed: int, mt_style: bool, n_epochs: int = 2):
    """Return a per-batch signature of the source stream one arm would see."""
    set_seed(seed)

    phi = FeatureExtractor(pretrained=True)
    classifier = Classifier(phi.out_dim, NUM_CLASSES)
    disc = CDANDisc(phi.out_dim, NUM_CLASSES)   # noqa: F841 - consumes RNG
    GradRevLayer()                              # parameterless, consumes none
    if mt_style:
        with torch.random.fork_rng(devices=[]):
            create_teacher(phi, classifier, NUM_CLASSES, "cpu")

    target_loader = loaders.target_three_view if mt_style else loaders.target_plain
    signatures = []
    for epoch in range(1, n_epochs + 1):
        seed_loader_epoch(seed, epoch)
        if mt_style:
            loaders.target_three_view_dataset.set_epoch(epoch)
        for (src_img, src_lbl), _ in zip(loaders.source_train, target_loader):
            signatures.append((round(float(src_img.sum()), 4), int(src_lbl.sum())))
    return signatures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--seed", type=int, default=DataConfig.seed)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    paths = Paths(args.root) if args.root else Paths.from_env()
    cfg = DataConfig(seed=args.seed)
    loaders = build_loaders(
        load_cached_fbr_samples(paths.fbr_cache),
        load_ppd_samples(paths.plant_pathology, verbose=False),
        cfg,
    )

    print("Probing the source stream of both arms (no training)...")
    baseline = probe_source_stream(loaders, args.seed, mt_style=False, n_epochs=args.epochs)
    mean_teacher = probe_source_stream(loaders, args.seed, mt_style=True, n_epochs=args.epochs)

    print(f"  source batches compared: {len(baseline)}")
    if baseline == mean_teacher:
        print("  [OK] both arms see the same images in the same order.")
        print("       The accuracy gap is attributable to the consistency loss.")
        return

    first = next(i for i, (a, b) in enumerate(zip(baseline, mean_teacher)) if a != b)
    print(f"  [FAIL] streams diverge at batch {first}.")
    print(f"         cdane: {baseline[first]}")
    print(f"         mt   : {mean_teacher[first]}")
    print("  Check: model construction order, the fork around create_teacher,")
    print("  and the fork inside TargetThreeViewDataset.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
