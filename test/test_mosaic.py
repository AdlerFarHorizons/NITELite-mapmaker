'''Test file for mosaicking.
'''

import glob
import os
import unittest

import numpy as np

from nitelite_mapmaker import data
from nitelite_mapmaker import mosaic as mos


class TestFit(unittest.TestCase):

    def setUp(self):

        pattern = './test_data/220513-FH135/images/manually_referenced/*.tif'
        self.test_fps = glob.glob(pattern)

    def test_fit(self):

        # Construct
        reffed_images = [data.ReferencedImage.open(_) for _ in self.test_fps]
        mosaic = mos.Mosaic.from_referenced_images(reffed_images)

        mosaic.fit(reffed_images)
