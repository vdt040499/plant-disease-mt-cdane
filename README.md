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

## Alignment with published baselines

Two papers in `papers/` are referenced for their published numbers. Only one
of them describes the same experiment as this repository.

### Jeon et al., 2026 (`papers/10_1-s2.0-S1574954125005886-main.pdf`) — comparable

This is the source of FBR and of the 91.1 +/- 4.22% CDAN+E figure quoted
above. Every training-affecting choice in this repository was checked against
the paper's Section 3.2-3.3 and Table 1-2:

| Choice | Paper | This repository |
| --- | --- | --- |
| Backbone | ResNet-18, ImageNet-pretrained | same |
| Source domain | PVD, apple, 825 images, 3 classes | same dataset, same classes, same public source |
| Source split | 75 / 25 train / val, applied after FBR to the entire source set | same |
| Target/test ratio | PPD, 900 adaptation / 600 test — a 60 / 40 ratio | same ratio (1038 / 692 of 1730 filtered images) |
| Batch size | 64 | same |
| Optimizer | AdamW, lr 1e-3 | same |
| Schedule | CosineAnnealingLR | same |
| Adversarial run length / selection | 300 epochs, select on lowest val loss from epoch 250 | same |
| CDAN discriminator | 2 hidden layers, 512 then 256 units | same |
| Entropy weight (`+E`) | `1 + exp(-H(p))` | same |

Two differences that matter for interpretation:

- **Seeds.** The paper reports mean +/- std over 5 seeds; this repository
  reports a single seed (42). A standard deviation of 4.22 points on the
  paper's own runs means a one-seed delta of a few points is not evidence of
  being better or worse than the paper, only of being consistent or
  inconsistent with its reported range. Running `--arm cdane` under
  additional seeds is what would close that gap.
- **FBR background pool.** The paper draws its 900 target and 600 test images
  from a much larger PPD pool (~4900 images) and explicitly reserves the
  *remaining* images — a set disjoint from both target and test — as the
  source of background patches for FBR ("Section 3.1: *the remaining
  real-field images from these datasets were used for the proposed
  background augmentation method*"). This repository has no such third pool:
  `load_ppd_samples` keeps every single-label PPD image (1730 after
  filtering), and `split_ppd` divides all of it 60/40 into target and test.
  The FBR cache then draws its background patches from the target split
  itself (see the comment "Background images from the TARGET split only" in
  the FBR pre-compute cell) — the same 1038 images that are later used as
  the unlabeled domain-alignment/consistency set. The ratio matches the
  paper; the independence of the background pool from the target pool does
  not. Each target image is therefore used twice (as a background donor and
  as an adaptation sample), which the paper's design avoids. This has not
  been shown to bias the reported numbers, but it is a real deviation from
  the published protocol, not just a smaller dataset.

### Ilsever and Baz, 2024 (`papers/9_1-s2.0-S2772375524002181-main.pdf`) — not comparable

This paper is sometimes read as "CDAN+E combined with Mean Teacher on
ResNet-50" because it reports Mean Teacher results with a ResNet-50 backbone.
It contains no CDAN+E, no domain-adversarial component, no gradient reversal,
and no cross-domain setting of any kind — that reading is a category error,
not a nuance. It is a single-domain semi-supervised learning study: a
fraction of the labels in one dataset (PP2021TS) is withheld, and Mean
Teacher is asked to recover the gap.

| | Ilsever & Baz | This repository (CDAN+E + MT) |
| --- | --- | --- |
| Task | Semi-supervised learning, one domain, partial labels | Unsupervised domain adaptation, two domains, target fully unlabelled |
| Loss | `L_sup + w(t) * L_unsup` (no domain term) | `L_cls + L_cdan + w(t) * L_cons` |
| Backbone | ResNet-50 | ResNet-18 |
| Dataset | PP2021, 6 classes, ~17k images, one photographic domain | PVD (lab) -> PPD (field), 3 classes, ~2300 images total |
| Labelled / unlabelled split | Same distribution; 5% / 10% / 25% of one training set withheld | Different distributions entirely; target domain never labelled |
| Batch size / epochs | supervised ablation only: 32 / 90, early stopping. Mean Teacher itself: (10 labelled, 30 unlabelled) per batch, early stopping explicitly turned off | 64 / 300, fixed length, no early stopping |

The loss in this paper is structurally closest to this repository's
`train_mt_only` ablation (Mean Teacher without CDAN+E), not to the CDAN+E +
MT arm — and even that comparison would still cross backbone, dataset and
task boundaries. Its accuracy numbers cannot be placed in the results table
above; the paper is useful as motivation for using consistency
regularization under label scarcity, not as a numeric baseline. A numeric
comparison would require reproducing its exact setting (PP2021, ResNet-50,
its labelled-fraction protocol) as a separate experiment, not a variant of
the one in this repository.

**How the "25% / 10% / 5% / All" labelled fraction actually works (Section
3.1.2, Appendix A.2).** There is one training set, PP2021TS (15,546 images).
"All" means every image keeps its label — the fully-supervised reference.
"25%" means a stratified sample of 25% of that *same* training set keeps its
label; the remaining 75% is not discarded, it stays in training as the
unlabelled pool for `L_unsup` (confirmed by their own numbers: the "10%"
setting reports 13,992 unlabelled images, which is exactly the other 90% of
the 15,546-image training set). Labelled and unlabelled samples are drawn
from the *same* dataset and the *same* photographic domain — the split
simulates annotation scarcity, not a domain gap. This has no analogue on the
PVD side of this repository: PVD is 100% labelled by construction, and
"how much of PPD becomes FBR background vs. adaptation target" (previous
subsection) is an image-compositing decision, not a labelled-fraction split.
Point for point, this paper's protocol answers "how much labelled data does
Mean Teacher save you within one domain"; this repository's protocol answers
"does adding Mean Teacher on top of CDAN+E help once you already have zero
target labels and a domain gap." Different questions, not comparable numbers.

**On early stopping for the adversarial arms.** Both papers converge on the
same practical finding from different directions: Jeon et al. deliberately
fix adversarial training (CDAN+E, DANN) to 300 epochs with no early stopping,
because adversarial loss is not monotonic and a patience-based stop would
trigger on minimax noise rather than convergence (Section 3.2). Ilsever and
Baz reach the same conclusion for a different reason — they turn early
stopping off specifically for Mean Teacher because "the validation loss
increases in the first 5 epochs due to unsupervised weight ramp-up." Neither
paper uses early stopping on the method this repository's Pipeline 2 is
built from (CDAN+E) or resembles (Mean Teacher); a training-speed comparison
that adds early stopping to the CDAN+E + MT arm would not be reproducing
either paper's practice, and risks stopping on a spurious dip rather than
genuine convergence.

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
- Jeon et al. Bridging the Lab-to-Field gap in plant disease diagnosis through
  unsupervised domain adaptation enhanced by background recomposition.
  Ecological Informatics 93 (2026) 103579. Source of FBR and of the CDAN+E
  baseline this repository's Pipeline 1 is checked against.
- Ilsever and Baz. Consistency regularization based semi-supervised plant
  disease recognition. Smart Agricultural Technology 9 (2024) 100613.
  Single-domain Mean Teacher study, ResNet-50, no domain adaptation; see
  "Alignment with published baselines" for why its numbers are not a
  baseline for Pipeline 2.
