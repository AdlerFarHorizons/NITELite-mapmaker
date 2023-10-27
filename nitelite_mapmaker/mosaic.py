import numpy as np
import pandas as pd
import scipy
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from typing import Tuple

import cv2
from osgeo import gdal
import pyproj

from . import data


class Mosaic(data.Dataset):

    @classmethod
    def from_referenced_images(cls, filename, reffed_images, crs, bordersize=2):

        # Bounds
        x_bounds, y_bounds, pixel_width, pixel_height = data.get_containing_bounds(
            reffed_images,
            crs,
            bordersize=bordersize,
        )
        mosaic = Mosaic(
            filename,
            x_bounds,
            y_bounds,
            pixel_width,
            pixel_height,
            crs
        )

        return mosaic

    def incorporate_referenced_image(
        self,
        src: data.ReferencedImage,
        img: str = 'semitransparent_img_int',
    ):

        # Get existing data
        x_bounds, y_bounds = src.get_bounds(self.crs)
        dst_img = self.get_img(x_bounds, y_bounds)

        # Resize the image
        src_img = getattr(src, img)
        src_img_resized = cv2.resize(
            src_img[:, :, :self.n_bands],
            (dst_img.shape[1], dst_img.shape[0])
        )

        # Blend
        is_empty = (dst_img.sum(axis=2) == 0)
        dst_img = np.array([
            np.where(is_empty, src_img_resized[:, :, j], dst_img[:, :, j])
            for j in range(self.n_bands)
        ])
        dst_img = dst_img.transpose(1, 2, 0)

        self.save_img(dst_img, x_bounds, y_bounds)
