"""Networks, gradient reversal, CDAN+E conditioning and Mean Teacher parts."""

from plantuda.models.networks import CDANDisc, Classifier, FeatureExtractor
from plantuda.models.grl import GRL, GradRevLayer, domain_labels, get_lambda
from plantuda.models.cdan import cdan_entropy_weight, multilinear_map
from plantuda.models.mean_teacher import (
    consistency_weight,
    create_teacher,
    sigmoid_rampup,
    softmax_mse_loss,
    update_ema,
)

__all__ = [
    "FeatureExtractor",
    "Classifier",
    "CDANDisc",
    "GRL",
    "GradRevLayer",
    "get_lambda",
    "domain_labels",
    "multilinear_map",
    "cdan_entropy_weight",
    "sigmoid_rampup",
    "consistency_weight",
    "softmax_mse_loss",
    "create_teacher",
    "update_ema",
]
