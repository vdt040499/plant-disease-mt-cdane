# Plant Disease Diagnosis under Domain Shift

Unsupervised domain adaptation from laboratory leaf photographs to field
photographs, and a controlled measurement of what Mean Teacher adds on top of
CDAN+E.

A classifier trained on PlantVillage (leaves photographed against a plain
laboratory background) loses most of its accuracy on PlantPathology (leaves
photographed outdoors, with cluttered backgrounds and uncontrolled lighting).
No target labels are available. This repository implements two training arms
that address that shift and compares them under a protocol designed so the
difference between them is attributable to one loss term.

| Arm | Objective |
| --- | --- |
| CDAN+E | `L_cls + L_cdan` |
| CDAN+E + Mean Teacher | `L_cls + L_cdan + w(t) * L_cons` |

## Data

| Split | Domain | Images | Labels | Role |
| --- | --- | ---: | --- | --- |
| PVD train | Laboratory | 656 | Used | Supervised classification |
| PVD val | Laboratory | 219 | Used | Model selection only |
| PPD adaptation | Field | 1038 | Ignored | Domain alignment, consistency |
| PPD test | Field | 692 | Used once | Final evaluation |

Three classes: healthy, rust, scab. The source domain is capped at 300 images
per class (rust has only 275). Field rows labelled `multiple_diseases` are
dropped because the source domain has no matching class.

Source images are not used raw. Each is segmented with SAM and composited onto
a patch of a real field photograph, a step referred to here as FBR
(field-adaptive background recomposition). Background patches are drawn only
from the adaptation split, so no test-set imagery reaches training.

## Pipeline 1: CDAN+E

![CDAN+E pipeline](figures/pipeline-cdane.png)

Source and target images pass through one shared ResNet-18. The classification
head produces `L_cls` on labelled source images. The adversarial branch
conditions features on predictions through a multilinear map `f ⊗ p`, reverses
the gradient, and asks a discriminator to tell the domains apart. Because the
gradient is negated on its way back to the feature extractor, the extractor is
trained to make the two domains indistinguishable class by class rather than in
aggregate. The `+E` term weights each sample by `1 + exp(-H(p))`, so confident
predictions dominate the domain loss.

Two details in that figure are easy to get wrong and both fail silently:

- The multilinear map detaches the *predictions* but keeps the *feature*
  gradient. Detaching the features instead leaves the loss curves looking
  plausible while target accuracy collapses to chance.
- `lambda` rises from 0 to 1 on a schedule. Applying full adversarial pressure
  from step one destabilises the features before the classifier is usable.

## Pipeline 2: CDAN+E plus Mean Teacher

![CDAN+E plus Mean Teacher pipeline](figures/pipeline-cdane-mt.png)

Everything drawn in coral is what this arm adds. Each target image now yields
three views. The un-augmented view feeds the adversarial branch exactly as
before; two independently augmented views feed the student and a teacher. The
teacher is an exponential moving average of the student, receives no gradient,
and the disagreement between the two predictions becomes `L_cons`, a softmax
MSE weighted by `w(t)`.

`w(t)` is zero until epoch 100 and reaches 9.0 by epoch 150. The late ramp is
deliberate: `lambda` is still rising before epoch 100, and forcing consistency
while the feature space is being reshaped invites confirmation bias.

Three implementation details come from the reference implementations rather
than from any paper:

- The EMA update walks parameters only and never touches BatchNorm buffers.
  The teacher's buffers stay current because the teacher is kept in `.train()`
  mode and its own forward passes update them. ResNet-18 has 20 BatchNorm
  layers; leaving the teacher in `.eval()` makes it useless.
- The teacher is frozen at construction and its logits are additionally
  computed under `no_grad`.
- Model selection reads the teacher, not the student. The teacher is the
  product of the method; selecting on the student discards most of it.

## Experimental protocol

Both arms are run under one seed. A single seed cannot produce an error bar, so
the protocol compensates by removing every source of difference except the
consistency loss. Four invariants hold that pairing:

1. Modules are constructed in the same order in both arms, because each
   construction consumes the global RNG.
2. The teacher is built inside `torch.random.fork_rng`. It constructs two more
   modules; without the fork, every subsequent source augmentation would be out
   of phase with the baseline.
3. Both arms re-seed the shared DataLoader generator once per epoch with the
   same formula, so batch *k* of epoch *e* holds the same images in both runs.
   Without this, running one arm twice in a session gives different results at
   an identical seed.
4. The three-view dataset generates its augmented views inside its own forked
   RNG, seeded by `(seed, epoch, index)`. Views change every epoch, survive a
   resume, and never perturb the global stream.

`scripts/rng_probe.py` measures invariants 1 to 4 directly by replaying each
arm's RNG consumption and comparing a signature of every source batch. It takes
about a minute and performs no training. Run it before committing to a long
run; a failure means the comparison is void.

## Results

Seed 42, 300 epochs per arm, identical test split, identical selection rule
(lowest source validation loss from epoch 250 onwards).

| Method | Field test accuracy | Change |
| --- | ---: | ---: |
| No adaptation (source-only, earlier run) | 60.40% | reference |
| CDAN+E | 89.74% | +29.34 |
| CDAN+E + Mean Teacher | 95.81% | +6.07 over CDAN+E |

The Mean Teacher student's final weights reach 95.95%, close to the selected
teacher, which indicates the teacher's BatchNorm buffers are current.

How to read the 6.07 points. The correct statement is: on this seed, with every
other condition held fixed and verified, adding the consistency loss changed
accuracy by 6.07 points. It is not a claim that Mean Teacher improves accuracy
by 6.07 points in general. The published CDAN+E result under a comparable
setting is 91.1 +/- 4.22 percent; a seed-to-seed standard deviation of that
size means single-run differences of a few points can be noise. The paired
design is stronger evidence than two independent runs, and it is still n = 1.
Additional seeds are the next step for anyone extending this work.

## Repository layout

```
plantuda/
  config.py            Dataclasses describing a run: paths, data, training, Mean Teacher
  seeding.py           Seeding, the shared DataLoader generator, per-epoch re-seeding
  data/
    transforms.py      Stochastic train transform, deterministic evaluation transform
    datasets.py        PlantDiseaseDataset, TargetThreeViewDataset
    splits.py          Domain loading, stratified and seeded splits, FBR cache listing
    loaders.py         DataLoader construction and the Loaders bundle
  fbr/
    segment.py         SAM loading, leaf segmentation, mask refinement
    compose.py         Background patch sampling and alpha compositing
    cache.py           One composite per source image, written once and reused
  models/
    networks.py        ResNet-18 feature extractor, classifier, conditional discriminator
    grl.py             Gradient reversal and its lambda schedule
    cdan.py            Multilinear map and the entropy weight
    mean_teacher.py    Ramp, consistency loss, teacher construction, EMA update
  engine/
    train.py           The loop shared by both arms
    checkpoint.py      Resumable checkpoints, including teacher BatchNorm buffers
    evaluate.py        Test-set accuracy
scripts/
  prepare_data.py      Download both datasets
  build_fbr_cache.py   Pre-compute the FBR source images
  rng_probe.py         Verify that both arms see the same data stream
  train.py             Train one arm
  compare.py           Print the comparison table from stored results
  export_figures.py    Regenerate the figures from their HTML source
figures/
  pipeline-cdane-vs-mt.html   Source of truth for both diagrams
  pipeline-cdane.[png|svg]    Generated
  pipeline-cdane-mt.[png|svg] Generated
notebooks/
  Plant_Disease_MT_CDANE.ipynb   Original exploratory notebook
tests/
```

## Reproducing

```bash
pip install -r requirements.txt
export UDA_SAVE_DIR=/path/to/workspace     # datasets, caches, checkpoints, results

export KAGGLE_USERNAME=... KAGGLE_KEY=...  # required for PlantPathology
python scripts/prepare_data.py
python scripts/build_fbr_cache.py          # needs a GPU for SAM; resumable

python scripts/rng_probe.py                # verify the pairing before training
python scripts/train.py --arm cdane
python scripts/train.py --arm mt
python scripts/compare.py
```

Each arm writes `<UDA_SAVE_DIR>/results/<arm>_s<seed>.json` and skips itself if
that file exists; pass `--force` to re-run. Checkpoints are written every ten
epochs and resumed automatically, so an interrupted session can be restarted
with the same command. A full run is 300 epochs per arm.

For a fast end-to-end check, shorten every schedule so the consistency term is
actually exercised:

```bash
python scripts/train.py --arm mt --epochs 4 --min-select 3 \
    --cons-start 1 --cons-ramp-len 2 --log-every 1 --force
```

## Figures

`figures/pipeline-cdane-vs-mt.html` is the single source for both diagrams. It
is a self-contained HTML file: open it in a browser to view, edit the inline
SVG to change a label or a box, then regenerate the exported files.

```bash
python scripts/export_figures.py           # writes .svg and 2x .png for each diagram
python scripts/export_figures.py --scale 3 # print resolution
```

The README references the generated PNG files by path, so a regenerated figure
appears here with no edit to this document. PNG is used for display because it
carries the intended typography everywhere; the SVG files are for LaTeX and
vector editing. PNG export drives Playwright when installed and otherwise a
local Chrome or Chromium in headless mode.

## Notes

Datasets, checkpoints and caches live under `UDA_SAVE_DIR` and are never
committed. The Kaggle credentials for the PlantPathology download are read from
the environment; if a token was ever pasted into a notebook cell, rotate it and
treat the old one as public.

## References

- Long et al. Conditional Adversarial Domain Adaptation. NeurIPS 2018.
- Ganin and Lempitsky. Unsupervised Domain Adaptation by Backpropagation. ICML 2015.
- Tarvainen and Valpola. Mean teachers are better role models. NIPS 2017.
- Laine and Aila. Temporal Ensembling for Semi-Supervised Learning. ICLR 2017.
- Kirillov et al. Segment Anything. ICCV 2023.
