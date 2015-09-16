"""
The :mod:`data_balancing.data_conversion` module includes utilities to load and adapt datasets,
that takes advantage of :mod:`sklearn.datasets`.
"""

from .base import *

# datasets/* relay on .base
from datasets.coil_2000 import *


__all__ = ['get_data_home',
           'get_dataset_home',
           'check_fetch_data',
           'check_data',
           'check_label',
           'check_no_missing_data',
           'check_two_class_only',
           'check_data_type',
           'check_label_type',
           'fetch_coil_2000',
           'process_coil_2000',
           'check_coil_2000_data',
           'check_coil_2000_label',
           'convert_coil_2000',
           'load_coil_2000',
          ]



