"""Unsupervised domain adaptation for plant disease diagnosis.

PlantVillage (laboratory images, labelled) -> PlantPathology (field images,
unlabelled), with two training arms that share one code path:

    cdane : L_cls + L_cdan
    mt    : L_cls + L_cdan + w(t) * L_cons

The arms are designed to be *paired*: same seed, same weight initialisation,
same batch order, same test set, same model-selection rule. The only difference
is the consistency term. See `plantuda.engine.train` for the invariants that
keep that pairing valid.
"""

__version__ = "1.0.0"
