"""
The :mod:`data_balancing.data_loading` module includes utilities to load datasets,
that takes advantage of :mod:`sklearn.datasets`.
"""

# from .base import load_diabetes
# from .base import load_digits
# from .base import load_files
# from .base import load_iris
# from .base import load_linnerud
# from .base import load_boston
from sklearn.datasets.base import get_data_home
from sklearn.datasets.base import clear_data_home
# from .base import load_sample_images
# from .base import load_sample_image
# from .covtype import fetch_covtype
# from .mlcomp import load_mlcomp
# from .lfw import load_lfw_pairs
# from .lfw import load_lfw_people
# from .lfw import fetch_lfw_pairs
# from .lfw import fetch_lfw_people
# from .twenty_newsgroups import fetch_20newsgroups
# from .twenty_newsgroups import fetch_20newsgroups_vectorized
from sklearn.datasets.mldata import fetch_mldata, mldata_filename
# from .samples_generator import make_classification
# from .samples_generator import make_multilabel_classification
# from .samples_generator import make_hastie_10_2
# from .samples_generator import make_regression
# from .samples_generator import make_blobs
# from .samples_generator import make_moons
# from .samples_generator import make_circles
# from .samples_generator import make_friedman1
# from .samples_generator import make_friedman2
# from .samples_generator import make_friedman3
# from .samples_generator import make_low_rank_matrix
# from .samples_generator import make_sparse_coded_signal
# from .samples_generator import make_sparse_uncorrelated
# from .samples_generator import make_spd_matrix
# from .samples_generator import make_swiss_roll
# from .samples_generator import make_s_curve
# from .samples_generator import make_sparse_spd_matrix
# from .samples_generator import make_gaussian_quantiles
# from .samples_generator import make_biclusters
# from .samples_generator import make_checkerboard
# from .svmlight_format import load_svmlight_file
# from .svmlight_format import load_svmlight_files
# from .svmlight_format import dump_svmlight_file
# from .olivetti_faces import fetch_olivetti_faces
# from .species_distributions import fetch_species_distributions
# from .california_housing import fetch_california_housing

__all__ = ['clear_data_home',
           # 'dump_svmlight_file',
           # 'fetch_20newsgroups',
           # 'fetch_20newsgroups_vectorized',
           # 'fetch_lfw_pairs',
           # 'fetch_lfw_people',
           'fetch_mldata',
           # 'fetch_olivetti_faces',
           # 'fetch_species_distributions',
           # 'fetch_california_housing',
           # 'fetch_covtype',
           'get_data_home',
           # 'load_boston',
           # 'load_diabetes',
           # 'load_digits',
           # 'load_files',
           # 'load_iris',
           # 'load_lfw_pairs',
           # 'load_lfw_people',
           # 'load_linnerud',
           # 'load_mlcomp',
           # 'load_sample_image',
           # 'load_sample_images',
           # 'load_svmlight_file',
           # 'load_svmlight_files',
           # 'make_biclusters',
           # 'make_blobs',
           # 'make_circles',
           # 'make_classification',
           # 'make_checkerboard',
           # 'make_friedman1',
           # 'make_friedman2',
           # 'make_friedman3',
           # 'make_gaussian_quantiles',
           # 'make_hastie_10_2',
           # 'make_low_rank_matrix',
           # 'make_moons',
           # 'make_multilabel_classification',
           # 'make_regression',
           # 'make_s_curve',
           # 'make_sparse_coded_signal',
           # 'make_sparse_spd_matrix',
           # 'make_sparse_uncorrelated',
           # 'make_spd_matrix',
           # 'make_swiss_roll',
           'mldata_filename',
          ]
