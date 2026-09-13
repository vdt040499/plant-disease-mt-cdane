"""Image transforms.

Two pipelines, and the difference between them is load-bearing:

TRAIN_TRANSFORM is stochastic (RandomCrop / flips / ColorJitter) and draws from
the global PyTorch RNG. VAL_TRANSFORM is deterministic and draws nothing.

The CDAN+E branch feeds target images through VAL_TRANSFORM in both arms. That
is what lets the Mean Teacher arm add two augmented views without changing the
adversarial branch's input by a single pixel.
"""

import torchvision.transforms as T

from plantuda.config import IMAGENET_MEAN, IMAGENET_STD

TRAIN_TRANSFORM = T.Compose(
    [
        T.Resize((256, 256)),      # leave room for the random crop
        T.RandomCrop(224),         # ResNet-18 input size
        T.RandomHorizontalFlip(),  # disease patterns are left-right invariant
        T.RandomVerticalFlip(),
        T.ColorJitter(             # lab lighting is fixed, field lighting is not
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

VAL_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),      # direct resize, no sampling
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)
