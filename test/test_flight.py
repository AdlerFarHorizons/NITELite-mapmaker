'''Test file for top-level mapmaking functionality.
'''

import os
import unittest

import numpy as np

from nitelite_mapmaker import observations

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
            camera_num=0,
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
            camera_num=0,
        )
        self.flight.update_metadata_with_cart_bounds()


class TestReferencedImage(unittest.TestCase):

    def test_constructor(self):

        generic_setup(self)
        self.flight.prep_metadata()
        self.flight.get_manually_georeferenced_filepaths(
            self.manually_referenced_dir,
            camera_num=0,
        )

        # Get the index corresponding to our test image.
        reffed_fps = self.flight.metadata['manually_referenced_fp']
        reffed_fps = reffed_fps.loc[reffed_fps.notna()]
        ind = reffed_fps.index[0]

        expected_obs = self.flight.get_referenced_observation(ind)
        x_bounds, y_bounds = expected_obs.get_cart_bounds()

        actual_obs = observations.ReferencedImage(
            img_int=self.obs.img_int,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

        np.testing.assert_allclose(
            expected_obs.img,
            actual_obs.img,
        )
        np.testing.assert_allclose(
            actual_obs.dataset.RasterXSize,
            expected_obs.dataset.RasterXSize,
        )


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


class TestReferencedObservation(TestObservation):

    def setUp(self):

        generic_setup(self)
        self.flight.prep_metadata()
        self.flight.get_manually_georeferenced_filepaths(
            self.manually_referenced_dir,
            camera_num=0,
        )

        # Get the index corresponding to our test image.
        reffed_fps = self.flight.metadata['manually_referenced_fp']
        reffed_fps = reffed_fps.loc[reffed_fps.notna()]
        ind = reffed_fps.index[0]

        self.obs = self.flight.get_referenced_observation(ind)

    def test_get_latlon_bounds(self):

        lon_bounds, lat_bounds = self.obs.latlon_bounds

        assert lon_bounds[1] > lon_bounds[0]
        assert lat_bounds[1] > lat_bounds[0]

    def test_get_cart_bounds(self):

        x_bounds, y_bounds = self.obs.cart_bounds

        assert x_bounds[1] > x_bounds[0]
        assert y_bounds[1] > y_bounds[0]

    def test_show_in_cart_crs(self):

        self.obs.show(crs='cartesian')
        self.obs.show(crs='pixel')

    def test_convert_pixel_to_cart(self):

        xs, ys = self.obs.get_cart_coordinates()
        pxs, pys = self.obs.get_pixel_coordinates()

        actual_xs, actual_ys = self.obs.convert_pixel_to_cart(pxs, pys)

        np.testing.assert_allclose(xs, actual_xs)
        np.testing.assert_allclose(ys, actual_ys)

    def test_convert_cart_to_pixel(self):

        xs, ys = self.obs.get_cart_coordinates()
        pxs, pys = self.obs.get_pixel_coordinates()

        actual_pxs, actual_pys = self.obs.convert_cart_to_pixel(xs, ys)

        np.testing.assert_allclose(pxs, actual_pxs)
        np.testing.assert_allclose(pys, actual_pys)


