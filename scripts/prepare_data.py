"""Download both datasets.

    python scripts/prepare_data.py

PlantVillage comes from GitHub, PlantPathology 2020 from Kaggle. The Kaggle
download needs credentials, read from the environment:

    export KAGGLE_USERNAME=...  KAGGLE_KEY=...

Credentials are never written into this repository. If you previously pasted a
token into a notebook cell, rotate it: anything committed to git should be
treated as public.
"""

import argparse
import glob
import os
import subprocess
import zipfile

import _bootstrap  # noqa: F401
from plantuda.config import Paths

PVD_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
PPD_COMPETITION = "plant-pathology-2020-fgvc7"


def download_plant_village(paths: Paths) -> None:
    target = paths.plant_village
    if target.exists() and any(target.iterdir()):
        print("PlantVillage already present, skipping.")
        return
    target.mkdir(parents=True, exist_ok=True)
    print("Cloning PlantVillage (shallow)...")
    subprocess.run(
        ["git", "clone", PVD_REPO, str(target), "--depth=1", "-q"], check=True
    )
    print("PlantVillage ready.")


def download_plant_pathology(paths: Paths) -> None:
    target = paths.plant_pathology
    if (target / "train.csv").exists():
        print("PlantPathology already present, skipping.")
        return
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise SystemExit(
            "Kaggle credentials missing. Set KAGGLE_USERNAME and KAGGLE_KEY, "
            "or download the competition data manually into " + str(target)
        )
    target.mkdir(parents=True, exist_ok=True)
    print("Downloading PlantPathology from Kaggle...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", PPD_COMPETITION, "-p", str(target)],
        check=True,
    )
    for archive in glob.glob(str(target / "*.zip")):
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        os.remove(archive)
    print("PlantPathology ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="override UDA_SAVE_DIR")
    args = parser.parse_args()

    paths = Paths(args.root) if args.root else Paths.from_env()
    paths.ensure()
    print(f"Root: {paths.root}")
    download_plant_village(paths)
    download_plant_pathology(paths)


if __name__ == "__main__":
    main()
