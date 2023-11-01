'''Test file for top-level mapmaking functionality.
'''

import os
import unittest

import numpy as np

from nitelite_mapmaker import observations, data
from . import test_data


def generic_setup(test_case):

    flight_name = '220513-FH135'
    test_case.metadata_dir = os.path.join('./test/test_data', flight_name)
    test_case.image_dir = os.path.join(
        './test/test_data',
        flight_name,
        'images/23085686'
    )
    test_case.manually_referenced_dir = os.path.join(
        './test/test_data',
        flight_name,
        'images/manually_referenced'
    )
    img_log_fp = os.path.join(test_case.metadata_dir, 'image.log')
    imu_log_fp = os.path.join(test_case.metadata_dir, 'OBC/PresIMULog.csv')
    gps_log_fp = os.path.join(test_case.metadata_dir, 'OBC/GPSLog.csv')

    test_case.flight = observations.Flight(
        image_dir=test_case.image_dir,
        img_log_fp=img_log_fp,
        imu_log_fp=imu_log_fp,
        gps_log_fp=gps_log_fp,
    )

    test_case.rng = np.random.default_rng(10327)


class TestPrepMetadata(unittest.TestCase):

    def setUp(self):

        generic_setup(self)

    def test_load_img_log(self):

        self.flight.load_img_log()

        # Check for no unnamed columns
        img_log_cols = self.flight.img_log_df.columns
        assert sum(['Unnamed' in column for column in img_log_cols]) == 0

    def test_load_imu_log(self):

        self.flight.load_imu_log()

    def test_load_gps_log(self):

        self.flight.load_gps_log()

    def test_get_combined_metadata(self):

        self.flight.load_img_log()
        self.flight.load_imu_log()
        self.flight.load_gps_log()

        self.flight.combine_logs()

    def test_prep_metadata(self):

        self.flight.prep_metadata()

    def test_get_manually_georeferenced_filepaths(self):
        '''TODO: Get rid of all mentions of "manually". Not necessary.'''

        self.flight.prep_metadata()
        fps = self.flight.get_manually_georeferenced_filepaths(
            self.manually_referenced_dir,
        )
        fp_camera_num = fps[fps.notna()].str.findall(
            r'(\d).tif'
        ).str[-1].astype(int)
        n_bad = (fp_camera_num != 0).sum()
        assert n_bad == 0

    def test_update_metadata_with_cart_bounds(self):
        self.flight.prep_metadata()
        fps = self.flight.get_manually_georeferenced_filepaths(
            self.manually_referenced_dir,
        )
        self.flight.update_metadata_with_cart_bounds()


class TestObservation(unittest.TestCase):

    def setUp(self):

        generic_setup(self)
        self.flight.prep_metadata()

        # Get the index corresponding to our test image.
        test_fn = '20220413_221313_1020286912_0_50_3.raw'
        self.obs = self.flight.get_observation(test_fn)

    def test_get_img(self):

        img = self.obs.get_img()

        assert img is not None

    def test_show(self):

        self.obs.show()

    def test_get_nonzero_mask(self):

        mask = self.obs.get_nonzero_mask()

        assert mask.shape == self.obs.img.shape[:2]

    def test_img_shape(self):

        assert self.obs.img_shape == self.obs.img.shape[:2]

    def test_semitransparent_img(self):

        np.testing.assert_allclose(
            self.obs.img,
            self.obs.semitransparent_img[:, :, :3],
        )


class TestReferencedObservation(TestObservation, test_data.TestReferencedImage):

    def setUp(self):

        super(TestObservation, self).setUp()

        generic_setup(self)
        self.flight.prep_metadata()
        self.flight.get_manually_georeferenced_filepaths(
            self.manually_referenced_dir,
        )

        # Get the index corresponding to our test image.
        reffed_fps = self.flight.metadata['manually_referenced_fp']
        reffed_fps = reffed_fps.loc[reffed_fps.notna()]
        ind = reffed_fps.index[0]

        self.reffed = self.flight.get_referenced_observation(ind)
        self.obs = self.reffed


class TestReferencedImageConstructionConsistency(unittest.TestCase):

    def setUp(self):

        generic_setup(self)
        self.flight.prep_metadata()
        self.flight.get_manually_georeferenced_filepaths(
            self.manually_referenced_dir,
        )

        # Get the index corresponding to our test image.
        reffed_fps = self.flight.metadata['manually_referenced_fp']
        reffed_fps = reffed_fps.loc[reffed_fps.notna()]
        ind = reffed_fps.index[0]

        self.reffed = self.flight.get_referenced_observation(ind)

    def test_constructor(self):

        x_bounds, y_bounds = self.reffed.cart_bounds

        actual_obs = data.ReferencedImage(
            img=self.reffed.img,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

        np.testing.assert_allclose(
            self.reffed.img,
            actual_obs.img,
        )
        np.testing.assert_allclose(
            actual_obs.dataset.RasterXSize,
            self.reffed.dataset.RasterXSize,
        )
        np.testing.assert_allclose(
            np.array(actual_obs.cart_bounds),
            np.array(self.reffed.cart_bounds),
        )

    def test_constructor_int_input(self):

        x_bounds, y_bounds = self.reffed.cart_bounds

        actual_obs = data.ReferencedImage(
            img=self.reffed.img_int,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

        # Tolerance according to lost information.
        np.testing.assert_allclose(
            self.reffed.img,
            actual_obs.img,
            atol=1 / 255
        )
        np.testing.assert_allclose(
            actual_obs.dataset.RasterXSize,
            self.reffed.dataset.RasterXSize,
        )
        np.testing.assert_allclose(
            np.array(actual_obs.cart_bounds),
            np.array(self.reffed.cart_bounds),
        )
