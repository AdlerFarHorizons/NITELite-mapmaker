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

    def test_from_search_regions(self):

        reffed_images = [data.ReferencedImage.open(_) for _ in self.test_fps]
        cart_bounds = np.array([_.cart_bounds for _ in reffed_images])
        search_coords = 0.5 * np.sum(cart_bounds, axis=2)
        search_radii = np.diff(cart_bounds, axis=2)[:, 0, :].max(axis=1)

        mosaic = mos.Mosaic.from_search_regions(
            self.fp,
            search_coords,
            search_radii,
        )

        # Cartesian coords are going to be large, unlike latlon coords
        assert np.abs(mosaic.x_bounds).max() > 1000

        assert mosaic.x_bounds[0] <= cart_bounds[:, 0, :].min()
        assert mosaic.x_bounds[1] >= cart_bounds[:, 0, :].max()
        assert mosaic.y_bounds[0] <= cart_bounds[:, 1, :].min()
        assert mosaic.y_bounds[1] >= cart_bounds[:, 1, :].max()
        
