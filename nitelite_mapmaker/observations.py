'''This module handles data from observations, e.g. NITELite flights.
'''

from typing import Tuple

import glob
import os


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
