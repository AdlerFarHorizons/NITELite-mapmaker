'''Test file for top-level mapmaking functionality.
'''

import os
import unittest

import numpy as np

from nitelite_mapmaker import observations


class TestPrepMetadata(unittest.TestCase):

    def setUp(self):

        flight_name = '220513-FH135'
        self.metadata_dir = os.path.join('./test/test_data', flight_name)
        self.image_dir = os.path.join('./test/test_data', flight_name,
                                      'images/23085686')
        image_log_fp = os.path.join(self.metadata_dir, 'image.log')
        imu_log_fp = os.path.join(self.metadata_dir, 'OBC/PresIMULog.csv')
        gps_log_fp = os.path.join(self.metadata_dir, 'OBC/GPSLog.csv')

        self.flight = observations.Flight(
            image_dir=self.image_dir,
            image_log_fp = image_log_fp,
            imu_log_fp = imu_log_fp,
            gps_log_fp = gps_log_fp,
        )

    def test_load_image_log(self):

        self.flight.load_image_log()

        # Check for no unnamed columns
        image_log_cols = self.flight.image_log_df.columns
        assert sum(['Unnamed' in column for column in image_log_cols]) == 0

    def test_load_imu_log(self):

        self.flight.load_imu_log()

    def test_load_gps_log(self):

        self.flight.load_gps_log()

    def test_get_combined_metadata(self):

        self.flight.get_combined_metadata()

    def test_prep_metadata(self):

        self.flight.prep_metadata()
