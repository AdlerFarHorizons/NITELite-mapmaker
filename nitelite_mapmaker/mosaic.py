import numpy as np
import pandas as pd
import scipy
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from typing import Tuple, Union

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
        img: str = 'img_int',
        fill_value: Union[str, int, float] = None,
    ):

        # Fill value defaults to values that would be opaque
        if fill_value is None:
            if 'int' in img:
                fill_value = 255
            else:
                fill_value = 1.

        # Get existing data
        x_bounds, y_bounds = src.get_bounds(self.crs)
        dst_img = self.get_img(x_bounds, y_bounds)

        # Resize the image
        src_img = getattr(src, img)
        src_img_resized = cv2.resize(
            src_img,
            (dst_img.shape[1], dst_img.shape[0])
        )

        # Blend
        # Doesn't consider empty in the final channel.
        is_empty = (dst_img[:, :, :self.n_bands - 1].sum(axis=2) == 0)
        blended_img = []
        for j in range(self.n_bands):
            try:
                blended_img_j = np.where(
                    is_empty,
                    src_img_resized[:, :, j],
                    dst_img[:, :, j]
                )
            # When there's no band information in the one we're blending,
            # fall back to the fill value
            except IndexError:
                blended_img_j = np.full(
                    dst_img.shape[:2],
                    fill_value,
                    dtype=dst_img.dtype
                )
            blended_img.append(blended_img_j)
        blended_img = np.array(blended_img).transpose(1, 2, 0)

        self.save_img(blended_img, x_bounds, y_bounds)
