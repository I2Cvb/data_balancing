"""
The :mod:`data_balancing.data_conversion` module includes utilities to load and adapt datasets,
that takes advantage of :mod:`sklearn.datasets`.
"""

from .base import get_data_home
from fetch.coil_2000 import fetch_coil_2000

__all__ = ['get_data_home',
           'fetch_coil_2000',
          ]


