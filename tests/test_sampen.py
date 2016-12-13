import numpy
import os
import unittest

from pyeeg import samp_entropy


class SampEnTests(unittest.TestCase):
    def test_sampen_against_predictable_sequence(self):
        data = numpy.asarray([10, 20] * 2000)
        self.assertAlmostEqual(
            samp_entropy(data, 2, 0.2),
            0.0,
            places=2
        )

    def test_sampen_against_original_c_test_data(self):
        """Use test data from
        http://www.physionet.org/physiotools/sampen/c/sampentest.txt
        """
        dir = os.path.dirname(__file__)
        file_path = os.path.join(dir, './demo_data/sampentest.txt')
        data = []
        with open(file_path, 'r') as file:
            for row in file:
                data.append(float(row.strip()))

        self.assertEqual(
            samp_entropy(numpy.asarray(data), 2, 0.2),
            2.1233284920357112
        )


if __name__ == '__main__':
    unittest.main()
