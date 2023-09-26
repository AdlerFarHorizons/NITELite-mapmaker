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
        xs_original_frame = np.linspace(0., 10., original.shape[1])
        ys_original_frame = np.linspace(0., 10., original.shape[0])
        da_original_frame = (
            (xs_original_frame[1] - xs_original_frame[0])
            * (ys_original_frame[1] - ys_original_frame[0])
        )
        integrated = da_original_frame.sum() * da_original_frame

        xs_original_frame_mesh, ys_original_frame_mesh = np.meshgrid(
            xs_original_frame,
            ys_original_frame,
        )

        (xs_resampled, ys_resampled), resampled = self.greffer.resample(
            [xs_original_frame_mesh, ys_original_frame_mesh],
            original,
        )

        # Check that the integrals match
        da_resampled = (
            (xs_resampled[1] - xs_resampled[0])
            * (ys_resampled[1] - ys_resampled[0])
        )
        integrated_resampled = resampled.sum() * da_resampled
        np.testing.assert_allclose(integrated, integrated_resampled, rtol=0.02)

        # But the arrays shouldn't be the same
        if original.shape == resampled.shape:
            is_matching = np.isclose(original, resampled)
            assert is_matching.sum() != is_matching.size

    def test_orientation(self):

        assert False
