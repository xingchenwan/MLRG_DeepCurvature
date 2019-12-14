import torch

from .kfac import KFACOptimizer
from .ekfac import EKFACOptimizer
from .swats import SWATS

# We use custom-built Adam that integrates Adam and AdamW
from .adam import Adam

# For SGD, we use the inbuilt pytorch SGD optimiser
from torch.optim import SGD
