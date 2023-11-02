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

        pattern = './test/test_data/220513-FH135/images/manually_referenced/*.tif'
        self.test_fps = glob.glob(pattern)

        test_dir = './test_data/220513-FH135/mosaic/temp.tif'
        os.makedirs(test_dir, exist_ok=True)
        self.fp = os.path.join(test_dir, 'temp.tif')

    def tearDown(self):
        if os.path.isfile(self.fp):
            os.remove(self.fp)

    def test_fit(self):

        # Construct
        reffed_images = [data.ReferencedImage.open(_) for _ in self.test_fps]
        mosaic = mos.Mosaic.from_referenced_images(self.fp, reffed_images)

        mosaic.fit(self.test_fps)
