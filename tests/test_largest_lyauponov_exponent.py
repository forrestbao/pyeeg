import numpy
import unittest

from pyeeg import LLE


class LLETests(unittest.TestCase):
    def test_largest_lyauponov_exponent(self):
        data = numpy.asarray([3, 4, 1, 2, 4, 51, 4, 32, 24, 12, 3, 45])

        self.assertAlmostEqual(
            LLE(data, 2, 4, 1, 1),
            0.18771136179353307,
            places=12
        )


if __name__ == '__main__':
    unittest.main()
