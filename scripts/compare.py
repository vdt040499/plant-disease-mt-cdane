"""Print the comparison table from stored results.

    python scripts/compare.py

Reads <root>/results/{cdane,mt}_s<seed>.json, so it works after a restart
without re-running anything.
"""

import argparse
import json

import _bootstrap  # noqa: F401
from plantuda.config import Paths, TrainConfig

PAPER_CDANE_FBR = 91.1
PAPER_CDANE_FBR_STD = 4.22


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    paths = Paths(args.root) if args.root else Paths.from_env()
    results = {}
    for arm in ("cdane", "mt"):
        path = paths.results / f"{arm}_s{args.seed}.json"
        if not path.exists():
            raise SystemExit(f"missing result: {path}. Run scripts/train.py --arm {arm}")
        results[arm] = json.loads(path.read_text())

    base = results["cdane"]["acc_selected"]
    mt = results["mt"]["acc_selected"]

    print("=" * 62)
    print(f"{'Method':<34}{'Test (field)':>14}{'Delta':>14}")
    print("-" * 62)
    print(f"{'CDAN+E':<34}{base:>13.2f}%{'-':>14}")
    print(f"{'CDAN+E + Mean Teacher':<34}{mt:>13.2f}%{mt - base:>+13.2f}%")
    print("=" * 62)
    print(f"seed = {args.seed} for both arms; identical test split and selection rule.")
    print(f"Mean Teacher student (final weights): {results['mt']['acc_student_final']:.2f}%")
    print()
    print("External reference, not the baseline of this experiment:")
    print(f"  Published CDAN+E with FBR (apple): {PAPER_CDANE_FBR} +- {PAPER_CDANE_FBR_STD}%")

    if abs(mt - base) < 1.0:
        print()
        print("NOTE: |delta| < 1 point at n=1. Report as 'no clear effect'.")


if __name__ == "__main__":
    main()
