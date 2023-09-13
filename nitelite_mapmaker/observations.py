'''This module handles data from observations, e.g. NITELite flights.
'''

from typing import Tuple

import glob
import os

import cv2
import numpy as np
import pandas as pd


class Flight:

    def __init__(
        self,
        image_dir: str,
        metadata_fp: str,
        img_shape: Tuple[int, int] = (1200, 1920),
        max_val: int = 4091,
    ):
        '''
        Args:
            image_dir: Directory containing raw image files.
            metadata_fp: Filepath to metadata file.
            img_shape: Shape of image.
            max_val: Maximum value of raw image data. This corresponds to
                the integer type. Defaults to 4091 = 2**12 - 1 (i.e. 12-bit).
                It's not clear why it's a 12 bit integer, but it is.
        '''

        self.image_dir = image_dir
        self.metadata_fp = metadata_fp
        self.img_shape = img_shape
        self.max_val = max_val

        # Get list of image filepaths.
        self.image_fps = glob.glob(os.path.join(self.image_dir, '*.raw'))

    def load_metadata(self, fp: str = None):

        if fp is None:
            fp = self.metadata_fp

        metadata = pd.read_csv(fp)

        # Handle any extra whitespace
        metadata.rename(columns=lambda x: x.strip(), inplace=True)

        self.metadata = metadata
        return metadata

    def get_rgb_img(self, fp: str, conversion_method: str = 'opencv'):

        # Load and reshape raw image data.
        raw_img = np.fromfile(fp, dtype=np.uint16)
        raw_img = raw_img.reshape(self.img_shape)

        # Get RGB image
        # Good method (does interpolation).
        if conversion_method == 'opencv':
            img = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2RGB)
        # Homebrew, understandable method.
        elif conversion_method == 'crude':
            # Since the raw files are bayer sampled
            # the red pixels are at "[0::2,0::2]" locations,
            # green are at "[0::2,1::2]" and "[1::2,0::2]",
            # and blue at "[1::2,1::2]"
            red = raw_img[0::2, 0::2]
            green1 = raw_img[0::2, 1::2]
            green2 = raw_img[1::2, 0::2]
            green_avg = 0.5 * (green1 + green2)
            blue = raw_img[1::2, 1::2]
            img = np.array([red, green_avg, blue, ]).transpose(1, 2, 0)
        else:
            raise ValueError('Invalid conversion method.')

        # Scale RGB image to 0 to 1 for each channel.
        img = img / self.max_val

        return img
