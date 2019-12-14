import torchvision
#if torchvision.__version__ == "0.2.1":
#    print("Older torchvision found")
#    from . import data
#else:
from. import data as data

from . import (
    methods,
    models,
    losses,
    utils,
)

__all__ = [
    'methods',
    'models',
    'data',
    'losses',
    'utils',
]
