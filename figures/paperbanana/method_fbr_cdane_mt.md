# Method: FBR + CDAN+E + Mean Teacher for lab-to-field plant disease diagnosis

## Overview

We address unsupervised domain adaptation (UDA) from a labelled **source domain**
(laboratory leaf images, PlantVillage) to an unlabelled **target domain**
(real-field images, Plant Pathology 2020). The framework has two stages:

1. **Field-adaptive Background Recomposition (FBR)**, an offline data
   augmentation that places source leaves on real-field backgrounds.
2. **Adversarial adaptation with CDAN+E combined with a Mean Teacher**, trained
   end to end with the objective
   $$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{cdan} + w(t)\,\mathcal{L}_{cons}.$$

## Stage 1: Field-adaptive Background Recomposition (FBR)

1. **Leaf segmentation.** A pretrained Segment Anything Model (SAM) segments the
   leaf using a single **center point prompt** (image encoder → prompt encoder →
   mask decoder).
2. **Morphological refinement.** The raw mask is cleaned by **dilation**,
   **erosion** and **Gaussian blurring**, giving a soft mask $M$.
3. **Background generation.** Unlabelled real-field images are **randomly
   cropped and rotated** to produce a background $B$.
4. **Alpha compositing.** $\tilde{x}_s = M \odot x_s + (1 - M) \odot B$, which gives a
   background-augmented source image.

## Stage 2: CDAN+E + Mean Teacher

**Networks.** The **student** consists of a **feature extractor** $\Phi$ (ResNet
CNN) and a **label classifier** $G$. A **domain discriminator** $D$ (MLP)
predicts source vs. target. The **teacher** ($\Phi'$, $G'$) has the same
architecture as the student and is never updated by gradients.

**Classification loss.** $\mathcal{L}_{cls}$: cross-entropy of the student on
labelled source images.

**Conditional adversarial loss (CDAN+E).**
- **Multilinear conditioning:** the discriminator input is $f \otimes g$, the
  outer product of the feature $f = \Phi(x)$ and the class prediction
  $g = \mathrm{softmax}(G(f))$.
- **Entropy conditioning (+E):** each sample is weighted by $1 + e^{-H(g)}$, so
  confident predictions count more.
- **Gradient Reversal Layer (GRL)** between $f \otimes g$ and $D$. It acts as the
  identity in the **forward pass** and multiplies the gradient by $-\lambda$ in
  the **backward pass**. $D$ learns to distinguish the domains; through the GRL,
  $\Phi$ learns **domain-invariant features** that confuse $D$.

**Consistency loss (Mean Teacher).** Each target image yields two augmented
views. The student sees view 1 and the teacher sees view 2 (**stop-gradient**).
$\mathcal{L}_{cons}$ is the **softmax-MSE** between their predictions, scaled by a
**ramp-up weight** $w(t)$.

**Backpropagation.** The feature extractor is updated by
$$\theta_\Phi \leftarrow \theta_\Phi - \eta\Big(\tfrac{\partial \mathcal{L}_{cls}}{\partial \theta_\Phi} - \lambda\,\tfrac{\partial \mathcal{L}_{cdan}}{\partial \theta_\Phi} + w(t)\,\tfrac{\partial \mathcal{L}_{cons}}{\partial \theta_\Phi}\Big),$$
where the sign $-\lambda$ comes from the GRL.

**EMA update.** After each step, $\theta' \leftarrow \alpha\,\theta' + (1-\alpha)\,\theta$.
The teacher is used for inference on the target domain.

---

## Figure layout specification

Keep the figure clean. **Only these formulas may appear in the figure:**
(1) $\tilde{x}_s = M \odot x_s + (1-M) \odot B$;
(2) $f \otimes g$ and $1 + e^{-H(g)}$;
(3) GRL: forward $\times 1$, backward $\times(-\lambda)$;
(4) the $\theta_\Phi$ update rule;
(5) $\theta' \leftarrow \alpha\theta' + (1-\alpha)\theta$;
(6) $\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{cdan} + w(t)\mathcal{L}_{cons}$.
Every other element uses a short technical label only (no numbers or
hyperparameters).

**Panel (a), top band: "Field-adaptive Background Recomposition (FBR)".**
Left to right with image thumbnails: lab leaf images → SAM block (image
encoder, prompt encoder with a red center point, mask decoder) → "Morphological
refinement" (dilation / erosion / Gaussian blur) → refined masks; in parallel,
real-field images → "Random crop & rotation" → backgrounds; both join at an
alpha-compositing node with formula (1) → background-augmented images.

**Panel (b), bottom band: "CDAN+E + Mean Teacher".**
- Inputs on the left: source images with labels (blue) and target images
  (green), the latter split into view 1 and view 2 by "augmentation".
- **Student**: $\Phi$ drawn as a CNN pyramid of feature maps, followed by $G$
  drawn as a small neural network of nodes and edges → $\mathcal{L}_{cls}$.
- **CDAN+E branch**: $f$ and $g$ meet at an ⊗ node ("multilinear conditioning")
  with an "entropy conditioning" node beside it, formula (2) → a clearly drawn
  magenta **GRL** block → $D$ drawn as an MLP of nodes → "source / target" →
  $\mathcal{L}_{cdan}$. A small inset next to the GRL shows formula (3) with a
  solid forward arrow and a dashed red backward arrow.
- **Teacher**: an identical CNN pyramid + node network in orange, with a
  snowflake "stop-gradient" icon. A dashed orange arrow from student to teacher
  labelled "EMA" with formula (5). Student (view 1) and teacher (view 2)
  outputs meet at $\mathcal{L}_{cons}$ ("softmax-MSE").
- **Total loss** ⊕ node on the right with formula (6).
- **Backpropagation**: dashed red arrows from the losses back into the student
  and $D$; the arrow from $\mathcal{L}_{cdan}$ flips sign at the GRL and is
  labelled $-\lambda$. Formula (4) is placed near $\Phi$. No arrow enters the
  teacher.
- Legend: solid = forward pass, dashed red = backpropagation, dashed orange =
  EMA update.
