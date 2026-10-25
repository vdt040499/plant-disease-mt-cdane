# Plant Disease Diagnosis Across Domains

Unsupervised domain adaptation (UDA) from lab leaf images to field images. A classifier trained on PlantVillage loses most of
its accuracy on field photographs, and no field labels exist to fix that directly. This repository combines three components
for that shift: field-adaptive background recomposition (FBR), conditional adversarial adaptation with entropy conditioning
(CDAN+E), and Mean Teacher (MT). It evaluates them on **five benchmarks (A–E)** taken from published plant-disease UDA studies,
each under that study's own protocol.

**Main results** (seed 42, accuracy %, FBR + CDAN+E + MT against the best published number under the same protocol):

| Benchmark | Source → target | Protocol | Ours | Published | Reference |
| --- | --- | --- | --: | --: | --- |
| A | PlantVillage apple → Plant Pathology 2020 | held-out 692 | **96.82** (R18) / **97.54** (R50) | 91.1 ± 4.22 (CDAN+E w/ FBR, R18) | Jeon et al. 2026 |
| B | PlantVillage apple → Plant Pathology 2020 | transductive 1730 | **98.50** | 72.31 (MSUN) | Wu et al. 2023 |
| C | PlantVillage potato → PlantDoc potato | held-out 106 | **67.14** (mean of 2 classes) | 52.68 (source only) | Mahapatra et al. 2026 |
| D | as B, uncapped 2550-image source | transductive 1730 | 98.50 (balanced source) / 77.86 (uncapped) | 94.79 (CBPUDA, withdrawn) | Huang and Tsai 2023 |
| E | PlantVillage apple → PlantDoc apple (C-PD) | transductive 511 | **80.23** | 71.45 (MSUN, apple-only) | Wu et al. 2023 |

All numbers are single-seed. Section 4.3 states what each comparison does and does not support; the main open question is
how much of each gap is FBR rather than CDAN+E or MT.

**Contents**

1. [Method](#1-method)
2. [Experimental setup](#2-experimental-setup)
3. [Results](#3-results)
4. [Discussion](#4-discussion)
5. [Conclusion](#5-conclusion)
6. [Reproducing](#reproducing)
7. [Appendix A: Alignment with published baselines](#appendix-a-alignment-with-published-baselines)

---

## 1. Method

Three training arms share the same backbone, data stream, optimiser and model selection; they differ only in the loss.

| Arm                   | Objective                        | Figure |
| --------------------- | -------------------------------- | --- |
| CDAN+E only           | `L_cls + L_cdan`                 | Fig. 1 |
| Mean Teacher only     | `L_cls + w(t) * L_cons`          | Fig. 2 |
| CDAN+E + Mean Teacher | `L_cls + L_cdan + w(t) * L_cons` | Fig. 3 |

Source images are never used raw. **FBR** segments each source leaf with SAM (center-point prompt, then dilate / erode /
blur refinement) and alpha-composites it onto a randomly cropped and rotated patch of a real field image from the
unlabelled target set. FBR runs once per source image and is cached; it precedes all three arms.

### 1.1 Pipeline 1: CDAN+E only

![CDAN+E pipeline](figures/pipeline-cdane.png)

**Fig. 1.** Source and target images pass through one shared ResNet. The classification head produces `L_cls` on labelled
source images. The adversarial branch conditions features on predictions through a multilinear map `f ⊗ p`, reverses the
gradient, and asks a discriminator to tell the domains apart. The `+E` term weights each sample by `1 + exp(-H(p))`, so
confident predictions dominate the domain loss.

### 1.2 Pipeline 2: Mean Teacher only

![Mean Teacher pipeline](figures/pipeline-mt-only.png)

**Fig. 2.** The same student network with the adversarial branch removed: no multilinear map, no gradient reversal, no
discriminator. Labelled source images give `L_cls`. Each target image yields two independently augmented views; one feeds
the student and the other the teacher. The teacher is an exponential moving average (EMA) of the student, and the softmax
MSE between their predictions is `L_cons`, weighted by `w(t)`. Because there is no `L_cdan`, the student never sees the
un-augmented target view.

This follows the Mean Teacher recipe of Tarvainen and Valpola, as applied to plant disease images by Ilsever and Baz (2024),
adapted to domain adaptation. It differs from that paper in three ways: labelled and unlabelled images come from different
domains, `L_cons` is computed on target images only, and the consistency weight ramps late (epoch 100 to 150, final value 9)
instead of over five epochs.

### 1.3 Pipeline 3: CDAN+E plus Mean Teacher

![CDAN+E plus Mean Teacher pipeline](figures/pipeline-cdane-mt.png)

**Fig. 3.** Each target image yields three views. The un-augmented view feeds the adversarial branch exactly as in Fig. 1;
two independently augmented views feed the student and the teacher. The teacher receives no gradient, and the disagreement
between the two predictions is `L_cons`.

Three implementation details are load-bearing:

- The EMA update walks parameters only and never touches BatchNorm buffers. The teacher's buffers stay current because the
  teacher is kept in `.train()` mode and its own forward passes update them. ResNet-18 has 20 BatchNorm layers; leaving the
  teacher in `.eval()` makes it useless.
- The teacher is frozen at construction and its logits are additionally computed under `no_grad`.
- Model selection reads the teacher, not the student. The teacher is the product of the method; selecting on the student
  discards most of it.

---

## 2. Experimental setup

### 2.1 Benchmarks

Each benchmark reproduces the domain pair and evaluation protocol of one published study, so that its numbers can be set
against ours. *Held-out*: the target is split into an unlabelled adaptation set and a test set that is never used for
training, model selection or FBR backgrounds. *Transductive*: the model adapts on all target images and is scored on the
same images (labels are used only for scoring).

**Table 1.** Benchmarks, protocols and the published study each is compared with.

| | Domain pair | Protocol | Backbone | Compared with | Notebook |
| --- | --- | --- | --- | --- | --- |
| **A** | PlantVillage apple → Plant Pathology 2020 | held-out (692 test images) | ResNet-18 (ResNet-50 ablation) | [Jeon et al. 2026](https://www.sciencedirect.com/science/article/pii/S1574954125005886), Table 2 | [`A_apple_heldout.ipynb`](notebooks/A_apple_heldout.ipynb) |
| **B** | same | transductive (1730 images) | ResNet-50 | [Wu et al. 2023 (MSUN)](https://doi.org/10.34133/plantphenomics.0038), Table 6 | [`B_apple_transductive.ipynb`](notebooks/B_apple_transductive.ipynb) |
| **C** | PlantVillage potato → PlantDoc potato | held-out (106 test images) | ResNet-18 | [Mahapatra et al. 2026](<papers/5_Advancing plant disease classification using an attention-based CNN for intra-dataset and cross-dataset training.pdf>), Table 21 | [`C_potato_plantdoc.ipynb`](notebooks/C_potato_plantdoc.ipynb) |
| **D** | PlantVillage apple → Plant Pathology 2020 | as B, but the source is **all** 2550 PlantVillage apple images | ResNet-50 | [Huang and Tsai 2023 (CBPUDA)](https://arxiv.org/abs/2308.07017v1), Table IV — **withdrawn preprint**, reference only | [`D_apple_cbpuda.ipynb`](notebooks/D_apple_cbpuda.ipynb) |
| **E** | PlantVillage apple → PlantDoc apple, cropped by bounding box (C-PD) | transductive (511 crops) | ResNet-50 | [Wu et al. 2023 (MSUN)](https://doi.org/10.34133/plantphenomics.0038), Table 5 Apple column | [`E_apple_plantdoc.ipynb`](notebooks/E_apple_plantdoc.ipynb) |

### 2.2 Datasets

**Table 2.** Composition of the source, target and evaluation sets (class order healthy / rust / scab for apple). Target
labels are never used during adaptation.

| | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| Source (labelled, lab) | PlantVillage apple, 875 images (300 / 275 / 300) → 656 train / 219 val after FBR | same as A | PlantVillage potato early + late blight, 600 images (300 / class) → 450 train / 150 val | PlantVillage apple, **all 2550 images (1645 / 275 / 630)**, all used for training; no held-out val | same as A |
| Target (unlabelled, field) | Plant Pathology 2020, 1038 images (60% of the 1730 single-label images) | all 1730 images | PlantDoc potato, 106 images (50% of 212 after MD5 deduplication) | all 1730 images | PlantDoc apple boxes cut into crops (C-PD), **511 crops (239 / 138 / 134)** after manual filtering of 596 |
| Evaluation | 692 held-out images (40%) | the same 1730 target images | 106 held-out images (50%, stratified) | same as B | the same 511 crops |
| Classes | healthy, rust, scab | same | early blight, late blight | healthy, rust, scab | healthy, rust, scab |

- In A, B, C and E the source is capped at 300 images per class (apple rust has only 275), which keeps the classes roughly
  balanced. D removes the cap to match CBPUDA's split, so its source is 64.5% healthy and 10.8% rust while the target is
  29.8% healthy and 36.0% rust (Section 4.1).
- Plant Pathology rows labelled `multiple_diseases` are dropped (91 of 1821) because PlantVillage has no matching class.
- FBR background patches come only from unlabelled target images. In E the pool is the uncropped PlantDoc photos behind the
  target crops, reviewed by hand down to 103 outdoor photos with a short side of at least 256 px (Appendix A.5).

### 2.3 Implementation details

**Table 3.** Training configuration shared by all arms and benchmarks unless stated otherwise.

| Setting | Value |
| --- | --- |
| Backbone | ResNet-18 or ResNet-50, ImageNet-pretrained (Table 1) |
| Optimiser | AdamW, lr 1e-3, weight decay 0.01 |
| Schedule | CosineAnnealingLR, 300 epochs |
| Batch size | 64 (32 in E, whose target is small) |
| Model selection | lowest source validation loss from epoch 250, read from the teacher when MT is on (D: loss on the training images, since CBPUDA's split has no validation set) |
| CDAN discriminator | 2 hidden layers, 512 then 256 units; progressive GRL λ schedule |
| Entropy weight (`+E`) | `1 + exp(-H(p))` |
| Mean Teacher | EMA decay 0.99 (parameters only); `w(t) = 0` until epoch 100, then sigmoid ramp-up to 9 over 50 epochs; softmax-MSE consistency on target only |
| SAM | ViT-B or ViT-H (Table 6), center-point prompt; mask refinement by dilation, erosion and Gaussian blur |
| Seed | 42 (single seed for every run) |

Paired arms share the seed, the weight initialisation and the exact source-image stream. `scripts/rng_probe.py` verifies
this for A, so a difference between two paired arms is attributable to the loss term that differs.

### 2.4 Compared configurations

Following Jeon et al.'s ablation design, a configuration is named by the components it turns on:

- **Source only**: ResNet trained on raw PlantVillage, no FBR, no adaptation.
- **w/ FBR** (`src_fbr`): source only, but on FBR composites.
- **CDAN+E w/o FBR** (`cdane_raw`): adversarial adaptation on raw source images.
- **CDAN+E** (w/ FBR): Pipeline 1.
- **MT only** (w/ FBR): Pipeline 2.
- **CDAN+E + MT** (w/ FBR): Pipeline 3, the full method.

Not every configuration has been run on every benchmark; Table 9 lists what is missing.

---

## 3. Results

Accuracy in % on the evaluation set of Table 2, seed 42, 300 epochs. The numbers are the outputs saved in the notebooks from
the Colab runs and have not been re-run since the notebooks were split. Published numbers are copied from the cited tables;
the alignment of each protocol is detailed in Appendix A.

### 3.1 Comparison with published methods

#### Benchmark A: Plant Pathology 2020, held-out (Jeon et al.)

**Table 4.** Accuracy on the 692 held-out Plant Pathology images, ResNet-18. Jeon et al. report mean ± std over 5 seeds
(Table 2, Apple column; "UDA only" from Table 3). Ours is a single seed. Our CDAN+E row is the direct counterpart of
Jeon's CDAN+E w/ FBR and lands inside its reported range; adding Mean Teacher raises it further.

| Method | FBR | Accuracy |
| --- | :-: | --: |
| Baseline (source only) — Jeon et al. | – | 47.1 ± 6.54 |
| w/ FBR — Jeon et al. | ✓ | 57.0 ± 7.01 |
| DDC w/ FBR — Jeon et al. | ✓ | 57.7 ± 3.63 |
| DALN w/ FBR — Jeon et al. | ✓ | 59.3 ± 1.77 |
| DCORAL w/ FBR — Jeon et al. | ✓ | 61.0 ± 3.68 |
| DANN w/ FBR — Jeon et al. | ✓ | 72.4 ± 7.50 |
| CDAN+E w/o FBR (UDA only) — Jeon et al. | – | 88.4 ± 6.57 |
| CDAN+E w/ FBR — Jeon et al. | ✓ | 91.1 ± 4.22 |
| Real-to-Real (supervised on field images, upper bound) — Jeon et al. | – | 97.7 ± 0.77 |
| Source only (ours) † | – | 60.40 |
| CDAN+E (ours, SAM ViT-B / ViT-H) | ✓ | 89.74 / 94.08 |
| MT only (ours) | ✓ | 91.04 |
| CDAN+E + MT (ours, SAM ViT-B / ViT-H) | ✓ | **95.81** / **96.82** |
| CDAN+E + MT (ours, ResNet-50, SAM ViT-H) | ✓ | **97.54** |

† Hard-coded constant (`acc_on_B`) from an earlier full run; its configuration was not recorded.

#### Benchmark B: Plant Pathology 2020, transductive (MSUN)

**Table 5.** Per-class and overall accuracy on all 1730 Plant Pathology images, ResNet-50. Rows marked MSUN are copied from
Wu et al.'s Table 6 and use raw PlantVillage images; ours use FBR. Unlike every published method, our arms are balanced
across the three classes.

| Method | Scab | Rust | Healthy | Accuracy |
| --- | --: | --: | --: | --: |
| Baseline (w/o DA) — MSUN | 79.54 | 24.10 | 11.02 | 39.27 |
| DANN — MSUN | 85.12 | 30.26 | 28.71 | 50.04 |
| MRAN — MSUN | 81.28 | 52.57 | 70.87 | 67.45 |
| MSUN — MSUN | 87.32 | 57.71 | 73.86 | 72.31 |
| CDAN+E (ours, FBR) | 91.39 | 95.66 | 91.47 | 92.95 |
| CDAN+E + MT (ours, FBR) | **97.97** | **99.84** | **97.48** | **98.50** |

DAN, D-Coral, DAAN and DSAN fall between DANN and MRAN; the notebook prints all eight methods.

#### Benchmark E: PlantDoc apple (C-PD), transductive (MSUN)

**Table 6.** Accuracy on the 511 rebuilt C-PD apple crops, ResNet-50. MSUN rows are the Apple column of Wu et al.'s Table 5,
which comes from a single-species (apple-only, 3-class) run (Appendix A.5); MSUN reports one number per species.

| Method | Healthy | Rust | Scab | Accuracy |
| --- | --: | --: | --: | --: |
| Baseline (w/o DA) — MSUN | – | – | – | 67.38 |
| DANN — MSUN | – | – | – | 69.63 |
| DAN — MSUN | – | – | – | 70.29 |
| MSUN — MSUN | – | – | – | 71.45 |
| CDAN+E + MT (ours, FBR) | 74.06 | 91.30 | 79.85 | **80.23** |

D-Coral, DAAN, MRAN and DSAN fall between 65.51 and 69.06.

#### Benchmark C: PlantDoc potato, held-out (Mahapatra et al.)

**Table 7.** Per-class accuracy on the 106 held-out PlantDoc potato images. Mahapatra et al. (Table 21) train an attention
CNN on PlantVillage with no adaptation; its near-constant prediction (94.87 / 10.48) makes the per-class mean the fair single
number for this two-class problem.

| Method | Early blight | Late blight | Mean of 2 | Accuracy |
| --- | --: | --: | --: | --: |
| Attention CNN, source only — Mahapatra et al. | 94.87 | 10.48 | 52.68 \* | – |
| CDAN+E + MT (ours, ResNet-18, FBR with SAM ViT-H) | 64.29 | 70.00 | **67.14** | 66.98 |

\* Computed here; the paper reports per-class accuracy only.

#### Benchmark D: Plant Pathology 2020, CBPUDA split (reference only)

**Table 8.** Per-class and overall accuracy on all 1730 Plant Pathology images, ResNet-50. CBPUDA is a **withdrawn,
unreviewed preprint** (Appendix A.4). Our compared row is B's class-balanced source; the row with CBPUDA's uncapped
2550-image source collapses on rust (Section 4.1).

| Method | Healthy | Rust | Scab | Mean of 3 | Accuracy |
| --- | --: | --: | --: | --: | --: |
| ResNet-50 (w/o DA) — CBPUDA \* | 11.02 | 24.10 | 79.54 | 38.22 | 39.27 |
| DANN — CBPUDA \* | 28.71 | 30.26 | 85.12 | 48.03 | 50.04 |
| MRAN — CBPUDA \* | 70.87 | 52.57 | 81.28 | 68.24 | 67.45 |
| BCDM — CBPUDA | 97.09 | 1.93 | 88.51 | 62.51 | 59.65 |
| CBPUDA — CBPUDA | 94.96 | 92.92 | 96.79 | 94.89 | 94.79 |
| **CDAN+E + MT (ours, B, balanced 875-image source)** | 97.48 | 99.84 | 97.97 | 98.43 | **98.50** |
| CDAN+E + MT (ours, D, all 2550 source images) | 99.42 | 48.71 | 89.70 | 79.28 | 77.86 |

\* Identical, number for number, to MSUN's Table 6, so most likely copied from MSUN rather than re-run on CBPUDA's split.
Only BCDM and CBPUDA are the preprint's own runs.

### 3.2 Ablation study

**Table 9.** Contribution of each component. Each row differs from the one it is compared with by exactly one component, so
Δ is that component's contribution; Δ is given only where the two runs share backbone and SAM model.

| Exp | Configuration | FBR | CDAN+E | MT | Backbone | SAM | Accuracy | Δ |
| --- | --- | :-: | :-: | :-: | --- | --- | --: | --- |
| A | Source only | – | – | – | ResNet-18 | – | 60.40 † | |
| A | CDAN+E | ✓ | ✓ | – | ResNet-18 | ViT-B ‡ | 89.74 | |
| A | CDAN+E | ✓ | ✓ | – | ResNet-18 | ViT-H § | 94.08 | SAM ViT-B → ViT-H: **+4.34** |
| A | MT only | ✓ | – | ✓ | ResNet-18 | unconfirmed ‖ | 91.04 | |
| A | CDAN+E + MT | ✓ | ✓ | ✓ | ResNet-18 | ViT-B ‡ | **95.81** | + MT: **+6.07** |
| A | CDAN+E + MT | ✓ | ✓ | ✓ | ResNet-18 | ViT-H § | **96.82** | + MT: **+2.74**; SAM: +1.01 |
| A | CDAN+E + MT | ✓ | ✓ | ✓ | ResNet-50 | ViT-H § | **97.54** ¶ | backbone ResNet-18 → 50: +0.72 |
| B | CDAN+E | ✓ | ✓ | – | ResNet-50 | ViT-H ‡ | 92.95 | |
| B | CDAN+E + MT | ✓ | ✓ | ✓ | ResNet-50 | ViT-H ‡ | **98.50** | + MT: **+5.55** |
| C | CDAN+E + MT | ✓ | ✓ | ✓ | ResNet-18 | ViT-H | **66.98** | |
| D | CDAN+E + MT, all 2550 source images | ✓ | ✓ | ✓ | ResNet-50 | ViT-H \*\* | 77.86 | vs B (balanced source): **−20.64** †† |
| E | CDAN+E + MT | ✓ | ✓ | ✓ | ResNet-50 | ViT-H | **80.23** | |

**Not run yet** (these would complete the ablation): `src_fbr` and `cdane_raw` in A, B and C; MT only with a confirmed SAM in
A; every other arm in C; in D, the CDAN+E arm and a class-balanced sampler over the 2550 images (which would separate "more
source images" from "imbalanced source"); in E, `src_raw`, `src_fbr`, `cdane` and `mt_only`. Without them, the contribution of
FBR itself is not measured in this repository.

† Hard-coded constant (`acc_on_B`) from an earlier full run; configuration not recorded. ‡ SAM model as recorded in earlier
notes; B used the `fbr_h_cache` FBR cache, which by its name holds ViT-H masks. § Recorded in the earlier README; not present
in the saved notebook outputs. ‖ An earlier README listed 90.03 for "MT only, ViT-H", but 90.03 is the **student** accuracy
of this same run (teacher 91.04). ¶ Saved output 97.54; the earlier README said 97.64. \*\* Notebook default
(`SAM_MODEL_TYPE = 'vit_h'`, own cache `fbr_h_cache_cbp_full`); not recorded in the pasted output. †† Not a single-component
Δ: besides the source set, D selects the model on the training images themselves and uses its own freshly built FBR cache.

Findings:

- **Mean Teacher helps on top of CDAN+E in all three paired comparisons**: +6.07 (A, ViT-B), +2.74 (A, ViT-H), +5.55 (B). In
  A the paired runs share the seed, the initialisation and the exact source-image stream (verified by the RNG probe), so the
  difference is attributable to `L_cons`. The gain is smaller when the SAM masks are better (ViT-H): the less adapted the
  starting point, the more Mean Teacher adds.
- **Better SAM masks help more than a bigger backbone.** ViT-H over ViT-B: +4.34 (CDAN+E), +1.01 (CDAN+E + MT); ResNet-50
  over ResNet-18: +0.72.
- **A brackets the whole effect**: 60.40 without adaptation → 89.74 with FBR + CDAN+E → 95.81 with Mean Teacher added.
- **FBR's own share is taken from Jeon et al., not measured here.** In their ablation, CDAN+E gains +2.7 from FBR (88.4 →
  91.1) on the same pair; `cdane_raw` is the run that would measure it in this pipeline.

---

## 4. Discussion

### 4.1 Sensitivity to source class balance (B vs D)

B and D share the target, protocol and backbone and differ in the source: capped at 300 per class (98.50) or every
PlantVillage apple image (77.86). The drop is almost entirely rust (99.84 → 48.71; healthy 99.42, scab 89.70). The uncapped
model predicts 51 / 18 / 31% healthy / rust / scab on a target that is 30 / 36 / 34%.

The bias is already there at epoch 10, before `w(t)` leaves zero, so it comes from the classifier and CDAN+E. Mean Teacher
then locks it in: `L_cons` falls to 0.002 as the student agrees with a biased teacher. CDAN aligns conditional feature
distributions and implicitly assumes the two domains share class proportions, so this is the expected failure under label
shift (Zhao et al., ICML 2019). BCDM, one of CBPUDA's own runs on the same split, collapses the same way (rust 1.93%), so the
label shift is a property of the split, not only of this pipeline. More source images did not help because they came with a
label shift; the 300-per-class cap in A and B is load-bearing, not only a match to Jeon et al.

A possible reason CBPUDA is less affected, if its number is genuine: it classifies by distance to class prototypes on the
unit hypersphere, which has no bias term in which to store the source prior, and it has no self-training step that could
lock in a biased teacher. This is a hypothesis; the preprint does not test it.

### 4.2 Effect of protocol and target domain

**Table 10.** The same method across benchmarks.

| | A | B | C | E |
| --- | --: | --: | --: | --: |
| Best of ours | 97.54 (ResNet-50) / 96.82 (ResNet-18) | **98.50** | 66.98 | 80.23 |
| Same method, other protocol | held-out, ResNet-50: 97.54 | transductive: **98.50** (+0.96) | – | – |
| Difficulty of the target domain | easy (plain orchard photos) | easy | hard (internet photos, several leaves, clutter) | hard (internet photos; half the healthy crops are white-background stock images) |

- **Held-out vs transductive (A vs B).** The same configuration (ResNet-50, CDAN+E + MT) scores 97.54 held-out and 98.50
  transductive, so scoring on the images used for adaptation inflates the number by about one point. B exceeds even Jeon et
  al.'s supervised Real-to-Real bound (97.7) because Mean Teacher trains on the very images it is scored on.
- **Plant Pathology vs PlantDoc (A/B vs C/E).** The pipeline reaches the 90s on Plant Pathology but 67% on PlantDoc potato
  and 80% on PlantDoc apple. Models trained on PlantVillage alone score roughly 30–60% on PlantDoc in the cited papers (e.g.
  30.78% for MSUN's 27-class baseline). B and E share the source images, backbone, protocol and baseline paper; besides the
  target they differ in batch size (64 vs 32) and in the FBR background pool, which always comes from the target. 98.50
  against 80.23 therefore measures mostly, not purely, the harder target.

### 4.3 Per-class behaviour and error analysis

- **Class balance.** In B every class is above 91%, whereas every method in MSUN's Table 6 scores scab highest and leaves
  rust or healthy low. In C the published model collapses to one class (94.87 / 10.48) while this pipeline is balanced
  (64.29 / 70.00); the drop in early blight is therefore not a regression, only the absence of that collapse.
- **Healthy is the weak class in E** (74.06, against rust 91.30 and scab 79.85): 62 of the 101 errors are healthy leaves
  called diseased. A likely cause, not yet verified per crop, is that 114 of the 239 healthy crops (48%) are stock photos on a
  white background, against 3% of rust and 6% of scab crops. FBR puts every source leaf on an orchard background, so
  white-background leaves are unlike anything in training. The remaining confusion is scab called rust (23 crops),
  consistent with the red-spotted scab photos seen during filtering.

### 4.4 What the comparisons support

| Benchmark | Supported | Not yet supported |
| --- | --- | --- |
| A | Same protocol, backbone and hyper-parameters as Jeon et al.; our CDAN+E (89.74 / 94.08) is consistent with their 91.1 ± 4.22, and MT adds +2.74 to +6.07 in paired runs | Being better or worse than Jeon et al. by a few points (1 seed vs 5) |
| B | Under MSUN's protocol and backbone, FBR + CDAN+E (+ MT) is far more accurate than every method in Table 6 and balanced across classes; within the protocol, MT moves 92.95 → 98.50 in a paired comparison | Attributing the gap to MSUN to CDAN+E or MT rather than FBR (needs `cdane_raw`) |
| C | The full pipeline avoids the published model's single-class collapse (mean 67.14 vs 52.68) | Attributing that to adaptation rather than backbone and input size (needs `src_raw`) |
| D | Reference only: the preprint is withdrawn, and our compared run uses a different (balanced) source | Any claim against CBPUDA as a baseline |
| E | 8.8 points above MSUN's apple-only result under its protocol and backbone | Attributing that to CDAN+E or MT rather than FBR or the balanced source |

### 4.5 Limitations

- **Single seed.** Jeon et al. report ±4.22 for CDAN+E over five seeds, so differences of a few points are not yet evidence.
  Sampling error alone is about ±0.6 points at 98.5% on 1730 images, ±3.5 at 80% on 511, and ±9 at 67% on 106 (±13 per class).
- **FBR's contribution is not measured here** (missing `src_fbr`, `cdane_raw`, `src_raw` rows, Table 9).
- **FBR backgrounds come from target images.** In A they come from the adaptation split, which Jeon et al. keep disjoint from
  it (Appendix A.1); in B and E the pool is a subset of the evaluated images. None of the compared methods injects target
  pixels into source images this way.
- **SAM model not recorded for every run in A** (footnotes of Table 9).
- **Rebuilt target in E**: class counts match MSUN's to within two healthy crops, but not necessarily the same crops.

---

## 5. Conclusion

FBR + CDAN+E + Mean Teacher reaches 96.82% (ResNet-18) on Jeon et al.'s held-out Plant Pathology benchmark, against 91.1 ±
4.22 for their CDAN+E w/ FBR, and exceeds the published methods on MSUN's transductive Plant Pathology (98.50 vs 72.31) and
C-PD apple (80.23 vs 71.45) benchmarks. In every paired comparison, adding Mean Teacher to CDAN+E improves accuracy by 2.7–6.1
points. The pipeline is sensitive to label shift between source and target (77.86 with an imbalanced source), and PlantDoc
remains much harder than Plant Pathology. Multi-seed runs and the FBR-free ablation arms are the next steps needed to turn
these single-seed results into claims about the individual components.

---

## Reproducing

### The five benchmarks (Colab, GPU)

Open the notebook in Colab, set `SAVE_DIR` in section 2 to a folder on your Drive, and run the cells top to bottom. All
notebooks share the same structure (§2 configuration → §3 data → §4 SAM + FBR → loaders → models → training → results) and
the training set-up of Table 3. In section 2, `SAM_MODEL_TYPE` (`'vit_b'` | `'vit_h'`) is the one switch that picks the SAM
model; it also selects the FBR cache and the `runs_<sam>/` (A, B) or `results_<sam>/` (C) folder, so two SAM models never
overwrite or skip each other's results. [`Experiments_Summary.ipynb`](notebooks/Experiments_Summary.ipynb) rebuilds the result
tables and charts from recorded numbers (no training).

| To reproduce | Notebook | Settings | Cells to run |
| --- | --- | --- | --- |
| A: CDAN+E 89.74, CDAN+E + MT 95.81, MT only 91.04 | `A_apple_heldout.ipynb` | `SAM_MODEL_TYPE = 'vit_b'` (cache `fbr_cache`) | §3, §4, loaders, §5.1–5.2, then 5.3, 5.4 (helpers + `run_mt_only`), 5.5 (`train_cdane`, `run_mt`) |
| A: CDAN+E 94.08, CDAN+E + MT 96.82 | same | `SAM_MODEL_TYPE = 'vit_h'` (cache `fbr_h_cache`) | same |
| A: CDAN+E + MT, ResNet-50 (97.54) | same | `vit_h` | 5.7 (`run_mt_resnet50`) after the cells above |
| B: CDAN+E 92.95, CDAN+E + MT 98.50 | `B_apple_transductive.ipynb` | `vit_h`, ResNet-50 (set in `MSUN_RUNS`) | all definition cells, then §5.8 |
| C: CDAN+E + MT 66.98 | `C_potato_plantdoc.ipynb` | `vit_h`, `RUN_ARMS = ['cdane_mt']` | through §7, then §8 |
| D: CDAN+E + MT 77.86 | `D_apple_cbpuda.ipynb` | `vit_h`, `CBP_TRAIN_ON_ALL_SOURCE = True`, ResNet-50 (set in `CBP_RUNS`) | all definition cells, then §5.8 (first run builds a separate FBR cache for 2550 images) |
| E: CDAN+E + MT 80.23 | `E_apple_plantdoc.ipynb` | `vit_h`, `PROTOCOL = 'transductive'`, `BACKBONE = 'resnet50'`, `CPD_FILTER_DONE = True`; `cpd_apple/exclude.txt` and `fbr_bg_keep.txt` on Drive (FBR cache `fbr_vit_h_transductive_d42_n300_bgkeep`) | §1–§3.4, §4, loaders, §5–§7, then §8 with `RUN_ARMS = ['cdane_mt']` (default runs `src_raw`, `cdane`, `cdane_mt`) |

Notes. (1) The FBR caches already on Drive are reused, so SAM is not re-run; if a cache is missing it is rebuilt with the
same fixed seed. (2) Each run writes a JSON result, checkpoints every ten epochs and resumes after a Colab disconnect; delete
the JSON to force a fresh run. (3) The Kaggle token for Plant Pathology is read from Colab Secrets (`KAGGLE_API_TOKEN`) or the
environment, never from the notebook. (4) Cross-hardware or cross-version runs can differ slightly because some GPU
operations are not fully deterministic. (5) A's ablation was run across different sessions (the SAM model is not recorded in
the saved outputs of 5.3–5.7), which is why Table 9 flags which SAM is confirmed.

### The `plantuda` package (CPU / GPU, command line)

`plantuda/` is a separate, tested implementation of benchmark A's two paired arms (`cdane`, `mt`) with ResNet-18; it does not
implement the Mean-Teacher-only arm, and its numbers are not the ones in the tables above.

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

Each arm writes `<UDA_SAVE_DIR>/results/<arm>_s<seed>.json` and skips itself if that file exists; pass `--force` to re-run.
Checkpoints are written every ten epochs and resumed automatically. A full run is 300 epochs per arm. For a fast end-to-end
check, shorten every schedule so the consistency term is actually exercised:

```bash
python scripts/train.py --arm mt --epochs 4 --min-select 3 \
    --cons-start 1 --cons-ramp-len 2 --log-every 1 --force
```

### Figures

`figures/pipeline-cdane-vs-mt.html` is the single source for all three diagrams: a self-contained HTML file with inline SVG.
Edit it, then regenerate the exported files:

```bash
python scripts/export_figures.py           # writes .svg and 2x .png for each diagram
python scripts/export_figures.py --scale 3 # print resolution
```

The README embeds the generated PNG files by path, so a regenerated figure appears here with no edit to this document. The
SVG files are for LaTeX and vector editing. PNG export uses Playwright when installed and otherwise a local headless Chrome
or Chromium.

### Repository layout

```
plantuda/              Package implementing benchmark A's two arms (cdane, mt) with ResNet-18
  config.py            Dataclasses describing a run: paths, data, training, Mean Teacher
  seeding.py           Seeding, the shared DataLoader generator, per-epoch re-seeding
  data/                Transforms, datasets (incl. TargetThreeViewDataset), splits, loaders
  fbr/                 SAM segmentation, background compositing, the FBR cache
  models/              Networks, gradient reversal, CDAN+E, Mean Teacher primitives
  engine/              The loop shared by both arms, checkpoints, evaluation
scripts/               prepare_data, build_fbr_cache, rng_probe, train, compare, export_figures
figures/               pipeline-cdane-vs-mt.html (source) and the generated PNG / SVG diagrams
notebooks/             Colab notebooks A–E that produced the results (outputs saved inside), Experiments_Summary
papers/                Reference papers
tests/                 pytest suite, no GPU required
```

Datasets, checkpoints and caches live under `UDA_SAVE_DIR` / `SAVE_DIR` and are never committed. Credentials for the
PlantPathology download are read from the environment or Colab Secrets; if a token was ever pasted into a notebook cell,
rotate it and treat the old one as public.

---

## Appendix A: Alignment with published baselines

Three peer-reviewed papers are compared against. Jeon et al. describes the same experiment as A; Wu et al. (MSUN) runs the
same domain pair under a transductive protocol (B) and an apple-only transfer to PlantDoc (E); Mahapatra et al. supplies a
source-only number for the potato pair (C). A fourth, Huang and Tsai (CBPUDA), is a withdrawn preprint, listed for reference
only (D). Ilsever and Baz (2024) is cited only as the source of the Mean Teacher recipe in Pipeline 2; it is a single-domain
semi-supervised study and none of its numbers are compared.

### A.1 Benchmark A — Jeon et al., 2026 (`papers/10_1-s2.0-S1574954125005886-main.pdf`) — comparable

This is the source of FBR and of the 91.1 ± 4.22% CDAN+E figure. Every training-affecting choice was checked against the
paper's Sections 3.2–3.3 and Tables 1–2:

| Choice                             | Paper                                                           | This repository                                 |
| ---------------------------------- | --------------------------------------------------------------- | ----------------------------------------------- |
| Backbone                           | ResNet-18, ImageNet-pretrained                                  | same                                            |
| Source domain                      | PVD, apple, 825 images, 3 classes                               | same dataset and classes, 875 images            |
| Source split                       | 75 / 25 train / val, applied after FBR to the entire source set | same                                            |
| Target/test ratio                  | PPD, 900 adaptation / 600 test — a 60 / 40 ratio                | same ratio (1038 / 692 of 1730 filtered images) |
| Batch size                         | 64                                                              | same                                            |
| Optimizer                          | AdamW, lr 1e-3                                                  | same                                            |
| Schedule                           | CosineAnnealingLR                                               | same                                            |
| Adversarial run length / selection | 300 epochs, select on lowest val loss from epoch 250            | same                                            |
| CDAN discriminator                 | 2 hidden layers, 512 then 256 units                             | same                                            |
| Entropy weight (`+E`)              | `1 + exp(-H(p))`                                                | same                                            |

Two differences matter for interpretation:

- **Seeds.** The paper reports mean ± std over 5 seeds; this repository reports a single seed (42). A standard deviation of
  4.22 points means a one-seed delta of a few points is evidence only of being consistent or inconsistent with the reported
  range. Running `--arm cdane` under additional seeds would close that gap.
- **FBR background pool.** The paper draws its 900 target and 600 test images from a much larger PPD pool (~4900 images) and
  reserves the *remaining* images, disjoint from both target and test, as the source of FBR background patches (Section 3.1:
  *"the remaining real-field images from these datasets were used for the proposed background augmentation method"*). This
  repository has no such third pool: `load_ppd_samples` keeps every single-label PPD image (1730 after filtering), and
  `split_ppd` divides all of it 60/40 into target and test. The FBR cache then draws its background patches from the target
  split itself, the same 1038 images later used as the unlabelled alignment/consistency set. The ratio matches the paper; the
  independence of the background pool does not. Each target image is used twice (background donor and adaptation sample),
  which the paper's design avoids. This has not been shown to bias the numbers, but it is a real deviation from the published
  protocol.

### A.2 Benchmark B — Wu et al., 2023 (MSUN) — comparable under its own protocol

Wu, Fan, Luo, Choudhury, Tjahjadi and Hu. *From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease
Recognition in the Wild.* Plant Phenomics 5:0038 (2023). Its PVD → Plant-Pathology experiment uses the same two public
datasets and the same three apple classes, and Table 6 reports eight methods on it. MSUN itself is not re-implemented (the
authors publish no code); only the evaluation protocol is matched.

| Choice          | MSUN (PVD → PP)                                   | Notebook B                                         |
| --------------- | ------------------------------------------------- | -------------------------------------------------- |
| Target domain   | PPD, all images, unlabelled                       | all 1730 single-label PPD images, unlabelled       |
| Evaluation set  | the same target images (transductive)             | same                                               |
| Backbone        | ResNet-50, ImageNet-pretrained                    | same                                               |
| Source domain   | PVD apple, raw images                             | PVD apple through FBR (875 images, 656 train)      |
| Optimisation    | SGD, momentum 0.9, 200 epochs                     | AdamW, cosine, 300 epochs (this pipeline's own)    |

**The target set is very likely the same 1730 images.** MSUN's Table 2 lists PPD as 1604 / 370 / 1411 (scab / rust /
healthy). Those counts are identical to the corn table of the same paper, and Table 6 mislabels its scab column as "Gray
spot", so Table 2 cannot be taken at face value. Weighting Table 6's per-class accuracies by PP2020's single-label class
counts (592 / 622 / 516) reproduces the published "Average" column within 0.5 points for six of the eight methods (e.g. MSUN
72.66 vs 72.31, baseline 39.17 vs 39.27); weighting by Table 2's counts misses by up to 8.6 points. "Average" is therefore
read as overall accuracy on the 1730-image set. This is an inference from the numbers; the paper does not state it.

To state alongside the gap:

- **FBR is part of the pipeline; the eight methods in Table 6 use raw PVD images.** Jeon et al. report DANN at 72.4% with FBR
  on the same domain pair (different protocol), against 50.04% without FBR here. A large part of the gap is likely FBR.
- **FBR backgrounds come from evaluated images.** The background pool is the 1038-image adaptation split, a subset of the
  evaluation set under this protocol. Unlabelled use of target images is legitimate in transductive UDA, but none of the
  Table 6 methods injects target pixels into source images. PP2020's unlabelled Kaggle test images (`Test_*.jpg`) would give
  a pool disjoint from evaluation; not yet run.

### A.3 Benchmark C — Mahapatra et al., 2026 (`papers/5_Advancing plant disease classification using an attention-based CNN for intra-dataset and cross-dataset training.pdf`) — loosely comparable

Mahapatra, Panda, Dash and Sahu. *Advancing plant disease classification using an attention-based CNN for intra-dataset and
cross-dataset training.* Scientific Reports 16 (2026) 10925. Its cross-dataset experiment trains on PlantVillage potato and
tests on PlantDoc potato (early and late blight, the two classes both datasets share), with no adaptation.

| Choice        | Mahapatra et al.                                   | Notebook C                                            |
| ------------- | -------------------------------------------------- | ----------------------------------------------------- |
| Domain pair   | PlantVillage potato → PlantDoc potato, 2 classes   | same                                                  |
| PlantDoc test | 50% of PlantDoc (35 / 15 / 50 split)               | 50%, stratified, after MD5 deduplication (106 images) |
| PlantDoc rest | labelled training data for its intra-dataset runs  | unlabelled adaptation target                          |
| Model         | attention-based CNN, 150 × 150 input               | ResNet-18, 224 × 224, FBR (SAM ViT-H) + CDAN+E + MT   |
| Adaptation    | none                                               | yes                                                   |

- **The published model predicts almost every field image as early blight.** 94.87 / 10.48 is close to a constant classifier
  (100 / 0, mean 50); its high early-blight number is a by-product of that collapse.
- **Backbone, input resolution and adaptation all differ at once.** The notebook's `src_raw` arm (ResNet-18, raw PlantVillage,
  no adaptation, same split) would separate "better backbone" from "adaptation works"; it has not been run.
- **Ambiguity in the paper.** It says the model was "tested on the images of PlantDoc" without stating whether that means all
  222 images or only the 50% test split; this repository uses the 50% split so no evaluated image is ever trained on.

### A.4 Benchmark D — Huang and Tsai, 2023 (CBPUDA), withdrawn preprint — reference only

Huang and Tsai. *Contrastive Bi-Projector for Unsupervised Domain Adaption.* arXiv:2308.07017. Version 1 (14 Aug 2023)
reports PV → PP on the same three apple classes in Table IV. Version 2 (30 Dec 2023) **withdraws the paper**: *"Professor asks
to withdraw this paper on arxiv. This paper will upload again if the paper is published in the journal."* No journal version
was found. The paper is unreviewed, publishes no code, and its numbers are cited from v1 for reference only.

| Choice          | CBPUDA (§IV-A, IV-B)                                   | B (compared run)                          | D (CBPUDA's split)                         |
| --------------- | ------------------------------------------------------ | ----------------------------------------- | ------------------------------------------ |
| Source domain   | all PV apple, 1645 / 275 / 630 (healthy / rust / scab) | 875 images, 300 / 275 / 300, through FBR  | all 2550 images, through FBR               |
| Target domain   | PP2020 without `multiple_diseases`: 516 / 622 / 592    | same 1730 images                          | same                                       |
| Evaluation set  | the target itself (transductive)                       | same                                      | same                                       |
| Backbone        | ResNet-50, ImageNet-pretrained                         | same                                      | same                                       |
| Optimisation    | SGD, lr 1e-4 backbone / 5e-3 heads, batch 36, 10 000 iterations | AdamW, cosine, batch 64, 300 epochs | same as B                          |
| Model selection | not stated                                             | source val loss from epoch 250            | loss on the training images from epoch 250 |

**Why the compared row is B, not D.** Under CBPUDA's uncapped split, the source's label shift drives this pipeline into a
rust collapse (Section 4.1). B keeps every other choice of the protocol and differs only in using the class-balanced
875-image source, so the comparison is "this pipeline on a balanced source" against "CBPUDA on the full source", and has to
be stated that way. A class-balanced sampler over all 2550 images would put both sides on the same source images; not run.

- Table IV orders its columns healthy / rust / scab, the reverse of MSUN; the values themselves agree.
- **CBPUDA's "AVG" matches neither definition.** From its per-class values, the mean of three classes is 94.89 and the accuracy
  over 516 / 622 / 592 images is 94.85, while the table gives 94.79 (BCDM: 62.51 / 59.94 vs 59.65). Rounding to 0.01 cannot
  explain the gap. The same check reproduces both of this repository's rows exactly (98.50, 77.86).
- **CBPUDA uses no FBR, and its model selection is not described.** The gap mixes FBR with the method, as in B.

### A.5 Benchmark E — Wu et al., 2023 (MSUN), apple on PlantDoc (C-PD) — comparable under its own protocol

MSUN's PVD → PD experiment (Table 5) transfers PlantVillage to C-PD, the version of PlantDoc in which every annotated box is
cut out as an image. Its headline number is a 27-class run over 13 species, but the per-species columns, including
**Apple**, come from separate single-species runs. The results section says so: *"From the experiments with single plant
species (i.e., PPD, CLD, TLD, and each plant species in C-PD), we observe that the transfer of single species is generally
better compared to the transfer of multiple plant species together in C-PD."* The numbers agree: in the Baseline row, the
six listed species alone would give an overall accuracy above the 30.78 printed under "Average" even if every other image
were wrong, so the columns cannot be per-species accuracies inside the 27-class model (assuming "Average" is overall
accuracy, as established in A.2). The Apple column is therefore the same 3-class task as this benchmark.

| Choice          | MSUN (PVD → C-PD, apple)                                        | Notebook E                                                     |
| --------------- | --------------------------------------------------------------- | -------------------------------------------------------------- |
| Target domain   | C-PD apple after removing wrong boxes: 241 / 138 / 134          | C-PD rebuilt from `PlantDoc-Object-Detection-Dataset`, filtered by hand: 239 / 138 / 134 |
| Evaluation set  | the same target images (transductive)                           | same, 511 crops                                                |
| Backbone        | ResNet-50, ImageNet-pretrained                                  | same                                                           |
| Source domain   | "20% of PVD"; apple 397 / 219 / 251 (Table 1), raw images       | 875 images (300 / 275 / 300) through FBR, as in A and B        |
| Optimisation    | SGD, momentum 0.9, 200 epochs                                   | AdamW, cosine, 300 epochs, batch 32                            |

**How the target was rebuilt.** MSUN does not publish its C-PD list. The notebook cuts every `Apple leaf` / `Apple rust leaf`
/ `Apple Scab Leaf` box out of PlantDoc's train and test annotations (596 crops: 247 / 178 / 171, after dropping one
duplicate image), rotates crops from phone photos to their EXIF orientation, and sorts them into one folder per class. Crops
were then removed by hand in two passes, recorded with reasons in `cpd_apple/exclude.txt` on Drive (85 crop IDs): non-apple
leaves, botanical drawings, boxes on fruit or buds, slivers and crops too small to show a leaf (first pass, 24); then a second
pass that brought rust and scab to exactly MSUN's counts (61). Healthy ends two crops short of MSUN.

**How the FBR background pool was chosen.** Backgrounds come from the uncropped PlantDoc photos behind the target crops (272
images), reviewed by hand and listed in `fbr_bg_keep.txt` (107 kept): outdoor photos in which foliage, branches, grass, soil
or sky take up a real part of the frame, with a short side of at least 256 px, since FBR upsamples a 30–70% patch of the
short side to 256 px. Removed: white or studio backgrounds, grey/blue/black backdrops, scanned paper, drawings, collages with
captions, indoor shots, single-leaf macro shots, large watermarks or banners. 103 of the 107 end up in the pool (the other
four have no crop left after filtering). The pool leans towards diseased-leaf photos (42 scab, 38 rust, 23 healthy by source
image) because most healthy PlantDoc photos are stock images on white; composites therefore often show diseased leaves
behind a PlantVillage leaf of another class.

To state alongside the gap:

- **FBR is part of the pipeline; MSUN uses raw PVD images.** `src_raw` is the row to set against MSUN's 67.38 baseline, and
  `src_fbr` separates FBR from CDAN+E and MT; neither has been run.
- **Different source sets.** MSUN's apple source is skewed towards healthy (46% of 867); ours is balanced, which D shows
  matters for this pipeline. MSUN's own description is also inconsistent: Table 1 lists 219 rust images, more than 20% of
  PlantVillage's 275.
- **FBR backgrounds come from evaluated images**, as in B; `PROTOCOL = 'heldout'` in the notebook splits C-PD 50/50 by source
  photo and keeps the pool on the target side.

---

## References

- Long et al. Conditional Adversarial Domain Adaptation. NeurIPS 2018.
- Ganin and Lempitsky. Unsupervised Domain Adaptation by Backpropagation. ICML 2015.
- Tarvainen and Valpola. Mean teachers are better role models. NIPS 2017.
- Laine and Aila. Temporal Ensembling for Semi-Supervised Learning. ICLR 2017.
- Kirillov et al. Segment Anything. ICCV 2023.
- Zhao, Des Combes, Zhang and Gordon. On Learning Invariant Representations for Domain Adaptation. ICML 2019. Why aligning
  feature distributions fails under label shift (Section 4.1).
- Jeon et al. Bridging the Lab-to-Field gap in plant disease diagnosis through unsupervised domain adaptation enhanced by
  background recomposition. Ecological Informatics 93 (2026) 103579.
  [Link](https://www.sciencedirect.com/science/article/pii/S1574954125005886) · [local PDF](papers/10_1-s2.0-S1574954125005886-main.pdf).
  Source of FBR and of benchmark A's baselines.
- Wu et al. From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease Recognition in the Wild. Plant
  Phenomics 5 (2023) 0038. [DOI](https://doi.org/10.34133/plantphenomics.0038). Source of benchmarks B and E's baselines.
- Mahapatra, Panda, Dash and Sahu. Advancing plant disease classification using an attention-based CNN for intra-dataset and
  cross-dataset training. Scientific Reports 16 (2026) 10925.
  [Local PDF](<papers/5_Advancing plant disease classification using an attention-based CNN for intra-dataset and cross-dataset training.pdf>).
  Source of benchmark C's source-only baseline.
- Ilsever and Baz. Consistency regularization based semi-supervised plant disease recognition. Smart Agricultural Technology
  9 (2024) 100613. [Link](https://www.sciencedirect.com/science/article/pii/S2772375524002181) · [local PDF](papers/9_1-s2.0-S2772375524002181-main.pdf).
  Source of the Mean Teacher recipe adapted in Pipeline 2 (single-domain study; its numbers are not used).
- Huang and Tsai. Contrastive Bi-Projector for Unsupervised Domain Adaption. arXiv:2308.07017v1 (2023), **withdrawn** in v2.
  [v1](https://arxiv.org/abs/2308.07017v1). Source of the CBPUDA / BCDM figures and the uncapped source split of benchmark D;
  reference only.
