import numpy
import unittest

from pyeeg import permutation_entropy


class PermutationEntropyTests(unittest.TestCase):
    def test_permutation_entropy(self):
        data = numpy.asarray([1, 2, 4, 5, 12, 3, 4, 5])

        self.assertEqual(
            permutation_entropy(data, 5, 1),
            2.0
        )


if __name__ == '__main__':
    unittest.main()
