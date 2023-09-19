'''Test file for top-level mapmaking functionality.
'''

import os
import unittest

import numpy as np

from nitelite_mapmaker import mapmaker


class TestGlobal(unittest.TestCase):

    def setUp(self):

        flight_name = '220513-FH135'
        self.metadata_dir = os.path.join('./test/test_data', flight_name)
        self.image_dir = os.path.join('./test/test_data', flight_name,
                                      'images/23085686')
        self.img_log_fp = os.path.join(self.metadata_dir, 'image.log')
        self.imu_log_fp = os.path.join(self.metadata_dir, 'OBC/PresIMULog.csv')
        self.gps_log_fp = os.path.join(self.metadata_dir, 'OBC/GPSLog.csv')

    def test_mapmake(self):

        mm = mapmaker.Mapmaker(
            image_dir=self.image_dir,
            img_log_fp=self.img_log_fp,
            imu_log_fp=self.imu_log_fp,
            gps_log_fp=self.gps_log_fp,
        )

        mm.mapmake()

    def test_prep(self):

        mm = mapmaker.Mapmaker(
            image_dir=self.image_dir,
            img_log_fp=self.img_log_fp,
            imu_log_fp=self.imu_log_fp,
            gps_log_fp=self.gps_log_fp,
        )

        mm.prep()

    def test_georeference(self):

        mm = mapmaker.Mapmaker(
            image_dir=self.image_dir,
            img_log_fp=self.img_log_fp,
            imu_log_fp=self.imu_log_fp,
            gps_log_fp=self.gps_log_fp,
        )

        mm.preprocess()
        mm.georeference()

    def test_construct_mosaic(self):

        mm = mapmaker.Mapmaker(
            image_dir=self.image_dir,
            img_log_fp=self.img_log_fp,
            imu_log_fp=self.imu_log_fp,
            gps_log_fp=self.gps_log_fp,
        )

        mm.preprocess()
        mm.georeference()
        mm.construct_mosaic()