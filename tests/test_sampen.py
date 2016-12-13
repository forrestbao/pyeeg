import numpy
import os
import unittest

from pyeeg import samp_entropy


class SampEnTests(unittest.TestCase):
    def setUp(self):
        dir = os.path.dirname(__file__)
        self.file_path = os.path.join(dir, './demo_data/sampentest.txt')

    def test_sampen(self):
        data = []
        with open(self.file_path, 'r') as file:
            for row in file:
                data.append(float(row.strip()))

        self.assertEqual(
            samp_entropy(numpy.asarray(data), 2, 0.2),
            2.1233284920357112
        )


if __name__ == '__main__':
    unittest.main()
