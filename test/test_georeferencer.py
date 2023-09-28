'''Test file for georeferencing
'''

import glob
import os
import unittest

import numpy as np

from nitelite_mapmaker import georeference, observations


class TestResample(unittest.TestCase):

    def setUp(self):

        self.rng = np.random.default_rng(1425)

    def get_rotated_coords(
        self,
        xs_img_frame,
        ys_img_frame,
        theta=np.pi / 4.,
        ll_coords=np.array([1., 2.]),
    ):

        xs_img_frame, ys_img_frame = np.meshgrid(
            xs_img_frame,
            ys_img_frame
        )

        xs = (
            ll_coords[0]
            + xs_img_frame * np.cos(theta)
            + ys_img_frame * np.sin(theta)
        )
        ys = (
            ll_coords[1]
            + xs_img_frame * np.sin(-theta)
            + ys_img_frame * np.cos(theta)
        )

        return xs, ys

    def check_integrals_and_values(
        self,
        xs_original_frame,
        ys_original_frame,
        itrans,
        integral_rtol=0.02,
        too_high_rtol=0.05,
    ):

        da_original_frame = (
            (xs_original_frame[1] - xs_original_frame[0])
            * (ys_original_frame[1] - ys_original_frame[0])
        )
        integrated = itrans.img.sum() * da_original_frame

        xs_resampled, ys_resampled = itrans.points_resampled

        # Check that the integrals match
        da_resampled = (
            (xs_resampled[1] - xs_resampled[0])
            * (ys_resampled[1] - ys_resampled[0])
        )
        integrated_resampled = itrans.img_resampled.sum() * da_resampled
        np.testing.assert_allclose(
            integrated,
            integrated_resampled,
            rtol=integral_rtol
        )

        # But the arrays shouldn't be the same
        if itrans.img.shape == itrans.img_resampled.shape:
            is_matching = np.isclose(itrans.original, itrans.img_resampled)
            assert is_matching.sum() != is_matching.size

        # Values >1 indicate that we're placing things weirdly--too many in
        # one spot, too few in another
        n_high = (itrans.img_resampled > 1.).sum() 
        assert n_high / itrans.img_resampled.size < too_high_rtol

    def test_uniform_dist(self):
        '''Check how random values between 0 and 1 are warmped
        '''

        original = self.rng.uniform(size=(4, 6))
        xs_original_frame = np.linspace(0., 1., original.shape[1])
        ys_original_frame = np.linspace(0., 1., original.shape[0])

        itrans = georeference.ImageTransformer(original)
        itrans.points = self.get_rotated_coords(
            xs_original_frame,
            ys_original_frame
        )

        points_resampled, resampled = itrans.resample()

        self.check_integrals_and_values(
            xs_original_frame,
            ys_original_frame,
            itrans,
        )

    # TODO: This does poorly, even though the flux preserved version does well.
    #       Return to this once we use a better resampler.
    # def test_uniform(self):
    #     '''Check how an array of 1s are warped.
    #     '''

    #     original = np.ones((100, 200))
    #     xs_original_frame = np.linspace(0., 1., original.shape[1])
    #     ys_original_frame = np.linspace(0., 1., original.shape[0])

    #     itrans = georeference.ImageTransformer(original)
    #     itrans.points = self.get_rotated_coords(
    #         xs_original_frame,
    #         ys_original_frame
    #     )

    #     points_resampled, resampled = itrans.resample(preserve_flux=False)

    #     # We want to ensure that the deviations from 1 are within an acceptable
    #     # amount.
    #     counts, _ = np.histogram(
    #         resampled.flatten(),
    #         [-0.5, 0.5, 1.5, np.inf],
    #     )
    #     f_correct = counts[1] / original.size
    #     assert f_correct > 0.95
    #     f_too_high = counts[2] / original.size
    #     assert f_too_high < 0.05
    #     f_empty = counts[0] / resampled.size
    #     assert f_empty < 0.6

    def test_orientation(self):

        # Image with one corner colored-in
        shape = (20, 30)
        original = np.zeros(shape)
        original[:shape[0] // 2, :shape[1] // 3] = 1

        xs_original_frame = np.linspace(0., 1., original.shape[1])
        ys_original_frame = np.linspace(0., 1., original.shape[0])

        itrans = georeference.ImageTransformer(original)
        itrans.points = self.get_rotated_coords(
            xs_original_frame,
            ys_original_frame
        )

        points_resampled, resampled = itrans.resample()

        # Check the orientation
        xs_high = itrans.points[0][original > 0.5]
        ys_high = itrans.points[1][original > 0.5]
        high_bounds = [
            [np.min(xs_high), np.max(xs_high)],
            [np.min(ys_high), np.max(ys_high)],
        ]
        xs_resampled_mesh, ys_resampled_mesh = np.meshgrid(
            points_resampled[0],
            points_resampled[1],
        )
        xs_high_resampled = xs_resampled_mesh[resampled > 0.5]
        ys_high_resampled = ys_resampled_mesh[resampled > 0.5]
        high_bounds_resampled = [
            [np.min(xs_high_resampled), np.max(xs_high_resampled)],
            [np.min(ys_high_resampled), np.max(ys_high_resampled)],
        ]
        atol = np.max([np.diff(xs_high_resampled), np.diff(ys_high_resampled)])
        np.testing.assert_allclose(
            high_bounds,
            high_bounds_resampled,
            atol=atol
        )

        self.check_integrals_and_values(
            xs_original_frame,
            ys_original_frame,
            itrans,
        )

    def test_actual(self):

        flight_name = '220513-FH135'
        self.metadata_dir = os.path.join('./test/test_data', flight_name)
        self.image_dir = os.path.join('./test/test_data', flight_name,
                                      'images/23085686')
        img_log_fp = os.path.join(self.metadata_dir, 'image.log')
        imu_log_fp = os.path.join(self.metadata_dir, 'OBC/PresIMULog.csv')
        gps_log_fp = os.path.join(self.metadata_dir, 'OBC/GPSLog.csv')

        flight = observations.Flight(
            image_dir=self.image_dir,
            img_log_fp=img_log_fp,
            imu_log_fp=imu_log_fp,
            gps_log_fp=gps_log_fp,
        )

        fp = self.rng.choice(glob.glob(os.path.join(self.image_dir, '*.raw')))
        original = flight.get_rgb_img(fp)

        xs_original_frame = np.linspace(0., 1., original.shape[1])
        ys_original_frame = np.linspace(0., 1., original.shape[0])

        itrans = georeference.ImageTransformer(original)
        itrans.points = self.get_rotated_coords(
            xs_original_frame,
            ys_original_frame
        )

        points_resampled, resampled, counts = itrans.resample(
            return_counts=True
        )

        self.check_integrals_and_values(
            xs_original_frame,
            ys_original_frame,
            itrans,
        )
