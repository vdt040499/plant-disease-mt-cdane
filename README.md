# Plant Disease Diagnosis Across Domains

Unsupervised domain adaptation (UDA) from lab leaf images (PlantVillage) to field images, where no field labels exist. The
pipeline combines field-adaptive background recomposition (**FBR**), conditional adversarial adaptation (**CDAN+E**) and
**Mean Teacher** (MT), and is evaluated on four benchmarks (A–D), each under the protocol of a published study.

**Main results** (seed 42, accuracy %, FBR + CDAN+E + MT):


|     | Original paper                                                                                                  | Source → target                            | Protocol          | Published method (backbone)  | Published   | Ours (FBR + CDAN+E + MT)          |
| --- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ----------------- | ---------------------------- | ----------- | --------------------------------- |
| A   | [Jeon et al. 2026](https://www.sciencedirect.com/science/article/pii/S1574954125005886), Ecol. Inform., Table 2 | PlantVillage apple → Plant Pathology 2020  | held-out 692      | CDAN+E w/ FBR (ResNet-18)    | 91.1 ± 4.22 | **96.82** (R18) / **97.54** (R50) |
| B   | [Wu et al. 2023 (MSUN)](https://doi.org/10.34133/plantphenomics.0038), Plant Phenomics, Table 6                 | PlantVillage apple → Plant Pathology 2020  | transductive 1730 | MSUN (ResNet-50)             | 72.31       | **98.50** (R50)                   |
| C   | [Huang and Tsai 2023 (CBPUDA)](https://arxiv.org/abs/2308.07017v1), arXiv, Table IV                             | PlantVillage apple → Plant Pathology 2020  | transductive 1730 | CBPUDA (ResNet-50)           | 94.79       | 98.50 (R50)                       |
| D   | [Wu et al. 2023 (MSUN)](https://doi.org/10.34133/plantphenomics.0038), Plant Phenomics, Table 5 (Apple)         | PlantVillage apple → PlantDoc apple (C-PD) | transductive 511  | MSUN, apple-only (ResNet-50) | 71.45       | **80.23** (R50)                   |


## 1. Method


| Arm                   | Objective                        | Diagram                                     |
| --------------------- | -------------------------------- | ------------------------------------------- |
| CDAN+E only           | `L_cls + L_cdan`                 | [Pipeline 1](figures/pipeline-cdane.png)    |
| Mean Teacher only     | `L_cls + w(t) * L_cons`          | [Pipeline 2](figures/pipeline-mt-only.png)  |
| CDAN+E + Mean Teacher | `L_cls + L_cdan + w(t) * L_cons` | [Pipeline 3](figures/pipeline-cdane-mt.png) |


![CDAN+E plus Mean Teacher pipeline](figures/pipeline-cdane-mt.png)

- **FBR** (before training, cached): each source leaf is segmented with SAM and alpha-composited onto a random patch of an
unlabelled target image.
- **CDAN+E**: a discriminator behind a gradient-reversal layer separates domains on the multilinear map `f ⊗ p`; each sample
is weighted by `1 + exp(-H(p))`.
- **Mean Teacher**: two augmented views of each target image feed the student and an EMA teacher; `L_cons` is their softmax
MSE, weighted by `w(t)`. In Pipeline 3 a third, un-augmented view feeds the adversarial branch. Adapted from Ilsever and
Baz (2024), with consistency on target images only and a late ramp (epoch 100 → 150, final weight 9).
- Load-bearing details: EMA updates parameters only and the teacher stays in `.train()` so its BatchNorm buffers update;
model selection reads the teacher, not the student.



## 2. Experimental setup

**Table 1.** Benchmarks. *Held-out*: the test images are never used for training, model selection or FBR backgrounds.
*Transductive*: adapt on all target images and score on the same images.


|     | Source (labelled)                                    | Target (unlabelled)                     | Evaluation   | Backbone  | Baseline paper        |
| --- | ---------------------------------------------------- | --------------------------------------- | ------------ | --------- | --------------------- |
| A   | PV apple, 875 (300 / 275 / 300), 75 / 25 train / val | PPD, 1038 (60%)                         | 692 held-out | ResNet-18 | Jeon et al., Table 2  |
| B   | same as A                                            | all 1730 PPD                            | same 1730    | ResNet-50 | MSUN, Table 6         |
| C   | PV apple, **all 2550** (1645 / 275 / 630), no val    | all 1730 PPD                            | same 1730    | ResNet-50 | CBPUDA, Table IV      |
| D   | same as A                                            | C-PD apple, 511 crops (239 / 138 / 134) | same 511     | ResNet-50 | MSUN, Table 5 (Apple) |


Classes are healthy / rust / scab. PPD = Plant Pathology 2020 with the 91
`multiple_diseases` rows dropped (1730 left). C-PD is PlantDoc with each annotated box cut out, rebuilt and filtered by hand
from 596 crops to match MSUN's class counts.

**Implementation.** AdamW (lr 1e-3, weight decay 0.01), cosine schedule, 300 epochs, batch 64 (32 in D). Selection on the
lowest source validation loss from epoch 250 (C: training loss, since it has no validation set). CDAN discriminator 512 → 256.
MT: EMA 0.99, sigmoid ramp-up. SAM ViT-B or ViT-H. Seed 42. Paired arms share the seed, initialisation and source-image
stream (checked by `scripts/rng_probe.py`), so a difference between them is attributable to the loss term that differs.

## 3. Results



### 3.1 Comparison with published methods

**Table 2.** Benchmark A, ResNet-18. Jeon et al.: mean ± std over 5 seeds (Table 2, Apple; "w/o FBR" from Table 3). Our
CDAN+E is the direct counterpart of theirs and falls inside their range.


| Method                                 | Jeon et al. | Ours              |
| -------------------------------------- | ----------- | ----------------- |
| Source only                            | 47.1 ± 6.54 | 60.40 †           |
| w/ FBR (no adaptation)                 | 57.0 ± 7.01 | –                 |
| DANN w/ FBR                            | 72.4 ± 7.50 | –                 |
| CDAN+E w/o FBR                         | 88.4 ± 6.57 | –                 |
| CDAN+E w/ FBR (SAM ViT-B / ViT-H)      | 91.1 ± 4.22 | 89.74 / 94.08     |
| MT only w/ FBR                         | –           | 91.04             |
| CDAN+E + MT w/ FBR (SAM ViT-B / ViT-H) | –           | **95.81 / 96.82** |
| Real-to-Real (supervised upper bound)  | 97.7 ± 0.77 | –                 |


**Table 3.** Benchmarks B, C (Plant Pathology, transductive) and D (C-PD apple, transductive), ResNet-50. Published methods
use raw PlantVillage images; ours use FBR. Per-class order healthy / rust / scab.


| Bench. | Method                                     | Per class             | Accuracy  |
| ------ | ------------------------------------------ | --------------------- | --------- |
| B      | Baseline — MSUN                            | 11.02 / 24.10 / 79.54 | 39.27     |
| B      | DANN — MSUN                                | 28.71 / 30.26 / 85.12 | 50.04     |
| B      | MSUN                                       | 73.86 / 57.71 / 87.32 | 72.31     |
| B      | CDAN+E (ours)                              | 91.47 / 95.66 / 91.39 | 92.95     |
| B      | CDAN+E + MT (ours)                         | 97.48 / 99.84 / 97.97 | **98.50** |
| C      | BCDM — CBPUDA                              | 97.09 / 1.93 / 88.51  | 59.65     |
| C      | CBPUDA                                     | 94.96 / 92.92 / 96.79 | 94.79     |
| C      | CDAN+E + MT (ours, all 2550 source images) | 99.42 / 48.71 / 89.70 | 77.86     |
| D      | Baseline — MSUN                            | –                     | 67.38     |
| D      | DANN — MSUN                                | –                     | 69.63     |
| D      | MSUN                                       | –                     | 71.45     |
| D      | CDAN+E + MT (ours)                         | 74.06 / 91.30 / 79.85 | **80.23** |




### 3.2 Ablation

**Table 4.** Each Δ changes exactly one component between runs that share backbone and SAM model.


| Exp | Configuration                | Backbone | SAM           | Accuracy | Δ                                      |
| --- | ---------------------------- | -------- | ------------- | -------- | -------------------------------------- |
| A   | CDAN+E                       | R18      | ViT-B ‡       | 89.74    |                                        |
| A   | CDAN+E + MT                  | R18      | ViT-B ‡       | 95.81    | + MT: **+6.07**                        |
| A   | CDAN+E                       | R18      | ViT-H §       | 94.08    | SAM B → H: +4.34                       |
| A   | CDAN+E + MT                  | R18      | ViT-H §       | 96.82    | + MT: **+2.74**; SAM B → H: +1.01      |
| A   | CDAN+E + MT                  | R50      | ViT-H §       | 97.54    | R18 → R50: +0.72                       |
| A   | MT only                      | R18      | unconfirmed ‖ | 91.04    |                                        |
| B   | CDAN+E                       | R50      | ViT-H ‡       | 92.95    |                                        |
| B   | CDAN+E + MT                  | R50      | ViT-H ‡       | 98.50    | + MT: **+5.55**                        |
| C   | CDAN+E + MT, uncapped source | R50      | ViT-H         | 77.86    | vs B: −20.64 (not single-component ††) |


† Constant from an earlier run, configuration not recorded. ‡ SAM model from earlier notes (B: cache `fbr_h_cache`).
§ From the earlier README, not in saved notebook outputs. ‖ 90.03, quoted earlier, is this run's *student*; the teacher is
91.04. †† C also selects on training loss and uses its own FBR cache.

- **MT adds +2.7 to +6.1 points on top of CDAN+E in every paired run**, more when the starting point is weaker (ViT-B masks).
- **Mask quality matters more than backbone size**: ViT-H over ViT-B gives up to +4.34, ResNet-50 over ResNet-18 +0.72.
- **Not run yet**: `src_fbr` and `cdane_raw` (A, B), MT only with a confirmed SAM (A), the other arms in D, a CDAN+E arm and a
class-balanced sampler in C. FBR's contribution is therefore only known from Jeon et al. (+2.7 for CDAN+E).



## 4. Discussion

- **Label shift breaks the pipeline (B vs C).** With every PlantVillage image, the source is 64.5% healthy and 10.8% rust
against 29.8% and 36.0% in the target, and rust recall drops from 99.84 to 48.71. The bias appears by epoch 10, before
`w(t)` leaves zero, so it comes from CDAN+E, which implicitly assumes shared class proportions (Zhao et al. 2019); MT then
locks it in (`L_cons` → 0.002). BCDM collapses the same way on that split (rust 1.93). The 300-per-class cap is essential.
- **Transductive scoring inflates by about one point**: the same configuration gives 97.54 held-out (A) and 98.50
transductive (B), above Jeon et al.'s supervised bound of 97.7.
- **PlantDoc is much harder than Plant Pathology** (internet photos, clutter, several leaves): 80% against the 90s. In D,
62 of 101 errors are healthy leaves called diseased; 48% of the healthy crops are white-background stock photos, unlike any
FBR composite.
- **Balanced predictions.** Every published method on B leaves at least one class far behind; ours stays above 91% per
class.

**Limitations.** (1) Single seed: Jeon et al. report ±4.22 over five seeds, and sampling error is about ±0.6 points on B,
±3.5 on D. (2) Gaps to MSUN mix FBR and source set with the method; the missing `cdane_raw` / `src_raw` runs
would separate them. (3) FBR backgrounds come from target images, which Jeon et al. keep
disjoint from the adaptation and test sets (see Appendix). (4) CBPUDA is a withdrawn, unreviewed preprint; C is reference
only, and our compared number uses the balanced source of B.

## 5. Conclusion

FBR + CDAN+E + Mean Teacher reaches 96.82% (ResNet-18) on Jeon et al.'s held-out benchmark against 91.1 ± 4.22 for their
CDAN+E w/ FBR, and exceeds the published results on MSUN's Plant Pathology (98.50 vs 72.31) and C-PD apple (80.23 vs 71.45)
benchmarks. Mean Teacher improves CDAN+E in every paired comparison. The method requires a class-balanced source, and
multi-seed runs plus the FBR-free arms are needed before attributing gains to individual components.

## Reproducing

**Notebooks (Colab, GPU).** Set `SAVE_DIR` in §2, then run top to bottom. `SAM_MODEL_TYPE` (`'vit_b'` | `'vit_h'`) selects the
SAM model, its FBR cache and output folder. Runs write a JSON result, checkpoint every ten epochs and resume after a disconnect.
The Kaggle token is read from Colab Secrets (`KAGGLE_API_TOKEN`) or the environment, never from the notebook.


| Result                                              | Notebook                                                             | Settings                                                                                                                                        |
| --------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| A (89.74 / 95.81 / 91.04; 94.08 / 96.82; R50 97.54) | `[A_apple_heldout.ipynb](notebooks/A_apple_heldout.ipynb)`           | `vit_b` (cache `fbr_cache`) or `vit_h` (`fbr_h_cache`); §5.3–5.5, R50 in §5.7                                                                   |
| B (92.95 / 98.50)                                   | `[B_apple_transductive.ipynb](notebooks/B_apple_transductive.ipynb)` | `vit_h`, §5.8                                                                                                                                   |
| C (77.86)                                           | `[C_apple_cbpuda.ipynb](notebooks/C_apple_cbpuda.ipynb)`             | `vit_h`, `CBP_TRAIN_ON_ALL_SOURCE = True`, §5.8                                                                                                 |
| D (80.23)                                           | `[D_apple_plantdoc.ipynb](notebooks/D_apple_plantdoc.ipynb)`         | `vit_h`, `PROTOCOL = 'transductive'`, `CPD_FILTER_DONE = True`, `exclude.txt` and `fbr_bg_keep.txt` on Drive, §8 with `RUN_ARMS = ['cdane_mt']` |


`[Experiments_Summary.ipynb](notebooks/Experiments_Summary.ipynb)` rebuilds the tables and charts without training.

`plantuda` **package (command line).** A tested re-implementation of A's two paired arms (`cdane`, `mt`, ResNet-18); its
numbers are not the ones above.

```bash
pip install -r requirements.txt
export UDA_SAVE_DIR=/path/to/workspace KAGGLE_USERNAME=... KAGGLE_KEY=...
python scripts/prepare_data.py && python scripts/build_fbr_cache.py   # SAM needs a GPU
python scripts/rng_probe.py                                            # verify pairing first
python scripts/train.py --arm cdane && python scripts/train.py --arm mt
python scripts/compare.py
```

Datasets, caches and checkpoints live under `$UDA_SAVE_DIR` and are never committed. Diagrams are generated from
`figures/pipeline-cdane-vs-mt.html` by `python scripts/export_figures.py`. See `CLAUDE.md` for a map of the code.

## Appendix: Alignment with published baselines

- **A — Jeon et al. (comparable).** Same backbone, optimiser, schedule, batch size, discriminator, entropy weight, 60 / 40
target / test ratio and epoch-250 selection. Two deviations: one seed instead of five (875 vs 825 source images is minor);
and the **FBR background pool**: Jeon et al. take backgrounds from PPD images disjoint from both target and test, whereas
here they come from the 1038-image target split itself, so each target image serves as both background donor and
adaptation sample.
- **B, D — MSUN (comparable under its protocol).** Same domain pair, transductive evaluation and ResNet-50; MSUN uses raw
source images and SGD for 200 epochs, and publishes no code. MSUN's Table 2 class counts are copied from its corn table,
but weighting its per-class accuracies by PP2020's counts (592 / 622 / 516) reproduces its "Average" within 0.5 points for
six of eight methods, so the target is very likely the same 1730 images. Table 5's Apple column comes from a single-species
run, so it is the same 3-class task as D. MSUN's apple source is 46% healthy, ours balanced.
- **C — CBPUDA (reference only).** arXiv:2308.07017 v1 was withdrawn by its authors in v2 and never published. Its baseline,
DANN and MRAN rows are identical to MSUN's Table 6, and its "AVG" matches neither the class mean nor the accuracy computed
from its per-class values.



## References

- Long et al. Conditional Adversarial Domain Adaptation. NeurIPS 2018.
- Ganin and Lempitsky. Unsupervised Domain Adaptation by Backpropagation. ICML 2015.
- Tarvainen and Valpola. Mean teachers are better role models. NIPS 2017.
- Kirillov et al. Segment Anything. ICCV 2023.
- Zhao et al. On Learning Invariant Representations for Domain Adaptation. ICML 2019.
- Jeon et al. Bridging the Lab-to-Field gap in plant disease diagnosis through unsupervised domain adaptation enhanced by
background recomposition. Ecological Informatics 93 (2026) 103579.
[Link](https://www.sciencedirect.com/science/article/pii/S1574954125005886) · [PDF](papers/10_1-s2.0-S1574954125005886-main.pdf)
- Wu et al. From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease Recognition in the Wild. Plant
Phenomics 5 (2023) 0038. [DOI](https://doi.org/10.34133/plantphenomics.0038)
- Ilsever and Baz. Consistency regularization based semi-supervised plant disease recognition. Smart Agricultural Technology
9 (2024) 100613. [PDF](papers/9_1-s2.0-S2772375524002181-main.pdf)
- Huang and Tsai. Contrastive Bi-Projector for Unsupervised Domain Adaption. arXiv:2308.07017v1 (2023), withdrawn.

