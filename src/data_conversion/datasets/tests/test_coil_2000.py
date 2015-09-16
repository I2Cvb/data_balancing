# Get the correct unittest library
try:
    import unittest2 as unittest
except ImportError:
    import unittest

from data_conversion.base import *
from data_conversion.datasets.coil_2000 import *

class Test_coil_2000(unittest.TestCase):

    def test_convert_coil_2000(self):
        convert_coil_2000()

def main():
    unittest.main()

if __name__ == '__main__':
    main()
