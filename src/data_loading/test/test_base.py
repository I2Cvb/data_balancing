# Get the correct unittest library
try:
    import unittest2 as unittest
except ImportError:
    import unittest

from .context import *
# from data_conversion.base import PROJECT_DATA_PATH
# from data_conversion.base import fetch_mldata

class Test_data_loading_base(unittest.TestCase):

    def test_raw_dataset_home(self):
        # get_data_home will point to a pre-existing folder
        data_home = get_raw_dataset_home()
        assert_equal(data_home, RAW_DATA_PATH)

        # ensure that the folder has been created
        assert_true(os.path.exists(data_home))

    def test_clean_dataset_home(self):
        # get_data_home will point to a pre-existing folder
        data_home = get_clean_dataset_home()
        assert_equal(data_home, CLEAN_DATA_PATH)

        # ensure that the folder has been created
        assert_true(os.path.exists(data_home))

def main():
    unittest.main()

if __name__ == '__main__':
    main()
