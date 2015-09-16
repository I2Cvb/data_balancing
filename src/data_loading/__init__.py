"""
The :mod:`data_balancing.data_loading` module includes utilities to load and
adapt datasets, that takes advantage of :mod:`sklearn.datasets`.
"""

from .base import *
from .data_constants import *

# datasets/* relay on .base
from datasets.coil_2000 import *


__all__ = ['get_raw_dataset_home',
           'get_clean_dataset_home',
           'load_coil_2000',
          ]



