import numpy
import unittest

from pyeeg import embed_seq


class EmbeddedSequenceTests(unittest.TestCase):
    def setUp(self):
        self.data = range(0, 9)

    def test_embedded_sequence_1_4(self):
        self.assertEqual(
            embed_seq(self.data, 1, 4).all(),
            numpy.asarray(
                [
                    [0., 1., 2., 3.],
                    [1., 2., 3., 4.],
                    [2., 3., 4., 5.],
                    [3., 4., 5., 6.],
                    [4., 5., 6., 7.],
                    [5., 6., 7., 8.]
                ]
            ).all()
        )

    def test_embedded_sequence_2_3(self):
        self.assertEqual(
            embed_seq(self.data, 2, 3).all(),
            numpy.asarray(
                [
                    [0., 2., 4.],
                    [1., 3., 5.],
                    [2., 4., 6.],
                    [3., 5., 7.],
                    [4., 6., 8.]
                ]
            ).all()
        )

    def test_embedded_sequence_4_1(self):
        self.assertEqual(
            embed_seq(self.data, 2, 3).all(),
            numpy.asarray(
                [
                    [0.],
                    [1.],
                    [2.],
                    [3.],
                    [4.],
                    [5.],
                    [6.],
                    [7.],
                    [8.]
                ]
            ).all()
        )


if __name__ == '__main__':
    unittest.main()
