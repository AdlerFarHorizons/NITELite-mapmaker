'''This module contains code for processing data.
'''

import cv2
import numpy as np

from .observations import Flight


class DataHandler:

    def __init__(self, flight: Flight):

        self.flight = flight

    def get_rgb_img(self, fp: str, conversion_method: str = 'opencv'):

        # Load and reshape raw image data.
        raw_img = np.fromfile(fp, dtype=np.uint16)
        raw_img = raw_img.reshape(self.flight.img_shape)

        # Get RGB image
        # Accurate method (does interpolation).
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
        img = img / self.flight.max_val

        return img
