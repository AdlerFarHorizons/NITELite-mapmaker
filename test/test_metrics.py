'''Test file for metric/scoring code
'''

import glob
import os
import unittest

import numpy as np

from nitelite_mapmaker import data, metrics


class TestMetrics(unittest.TestCase):

    def setUp(self):

        example_fp = (
            'test/test_data/220513-FH135/images'
            '/manually_referenced/Geo 836109848_1.tif'
        )
        self.image = data.Image.open(example_fp)

    def test_ccoeff(self):
        '''An offset of 1 pixel should produce a correlation coefficient
        of >0.75
        '''

        pad_width = 1
        padded_img = np.pad(
            self.image.img_int,
            ((pad_width, 0), (pad_width, 0), (0, 0)),
            constant_values=0
        )
        padded_img2 = np.pad(
            self.image.img_int,
            ((0, pad_width), (pad_width, 0), (0, 0)),
            constant_values=0
        )

        r = metrics.calc_ccoeff(padded_img, padded_img2)
        assert r > 0.75
