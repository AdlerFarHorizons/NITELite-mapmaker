'''Test file for georeferencing
'''

import os
import unittest

import numpy as np

from nitelite_mapmaker import georeference


class TestGlobal(unittest.TestCase):

    def setUp(self):

        self.greffer = georeference.GeoReferencer()
        self.rng = np.random.default_rng(1425)

    def test_conserved(self):

        original = self.rng.uniform(size=(4, 6))
        transformed = self.greffer.resample(original)

        # Should sum to the same
        np.testing.assert_allclose(original.sum(), transformed.sum())

        # But they shouldn't be the same
        is_matching = np.isclose(original, transformed)
        assert is_matching.sum() != is_matching.size
