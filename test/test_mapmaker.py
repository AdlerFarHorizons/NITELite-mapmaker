'''Test file for top-level mapmaking functionality.
'''

import os
import unittest

import numpy as np

from nitelite_mapmaker import mapmaker


class TestGlobal(unittest.TestCase):

    def setUp(self):

        flight_name = '220513-FH135'
        self.metadata_fp = os.path.join('./test/test_data', flight_name,
                                        'CollatedImageLog.csv')
        self.image_dir = os.path.join('./test/test_data', flight_name,
                                      'images/23085686')

    def test_mapmake(self):

        mm = mapmaker.Mapmaker()

        mm.mapmake()

    def test_load(self):

        mm = mapmaker.Mapmaker(
            image_dir=self.image_dir,
            metadata_fp=self.metadata_fp,
        )

        mm.load()

    def test_preprocess(self):

        mm = mapmaker.Mapmaker()

        mm.load()
        mm.preprocess()

    def test_georeference(self):

        mm = mapmaker.Mapmaker()

        mm.load()
        mm.preprocess()
        mm.georeference()

    def test_construct_mosaic(self):

        mm = mapmaker.Mapmaker()

        mm.load()
        mm.preprocess()
        mm.georeference()
        mm.construct_mosaic()