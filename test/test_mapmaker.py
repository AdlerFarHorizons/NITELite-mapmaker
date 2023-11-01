'''Test file for top-level mapmaking functionality.
'''

import glob
import os
import unittest

import numpy as np

from nitelite_mapmaker import mapmaker


class TestGlobal(unittest.TestCase):

    def setUp(self):

        flight_name = '220513-FH135'
        self.metadata_dir = os.path.join('./test/test_data', flight_name)
        self.root_image_dir = os.path.join(
            './test/test_data',
            flight_name,
            'images'
        )
        self.image_dir = os.path.join(self.root_image_dir, '23085686')
        self.referenced_image_dir = os.path.join(self.root_image_dir,
                                                 'manually_referenced')
        self.img_log_fp = os.path.join(self.metadata_dir, 'image.log')
        self.imu_log_fp = os.path.join(self.metadata_dir, 'OBC/PresIMULog.csv')
        self.gps_log_fp = os.path.join(self.metadata_dir, 'OBC/GPSLog.csv')

        self.mm = mapmaker.Mapmaker(
            image_dir=self.image_dir,
            img_log_fp=self.img_log_fp,
            imu_log_fp=self.imu_log_fp,
            gps_log_fp=self.gps_log_fp,
            referenced_dir=self.referenced_image_dir,
        )

    def test_prep(self):

        self.mm.prep()

    def test_fit(self):

        self.mm.prep()
        metadata = self.mm.flight.metadata
        train_fps = metadata.loc[
            metadata['manually_referenced_fp'].notna(),
            'manually_referenced_fp'
        ]
        self.mm.mosaicker.fit(train_fps)

    def test_predict(self):

        self.mm.mosaicker.load_fitted()
        self.mm.mosaicker.predict(self.mm.metadata)
