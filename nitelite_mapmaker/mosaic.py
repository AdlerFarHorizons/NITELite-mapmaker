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

from . import observations


class Mosaic:

    def __init__(
        self,
        filename: str,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        pixel_width: float,
        pixel_height: float,
        crs: pyproj.CRS,
        n_bands: int = 3,
    ):

        # Initialize an empty GeoTiff
        xsize = int(np.round((x_bounds[1] - x_bounds[0]) / pixel_width))
        ysize = int(np.round((y_bounds[1] - y_bounds[0]) / pixel_height))
        driver = gdal.GetDriverByName('GTiff')
        self.dataset = driver.Create(
            filename,
            xsize=xsize,
            ysize=ysize,
            bands=n_bands,
            options=['TILED=YES']
        )

        # Properties
        self.dataset.SetProjection(crs.to_wkt())
        self.dataset.SetGeoTransform([
            x_bounds[0],
            pixel_width,
            0.,
            y_bounds[1],
            pixel_height,
            0.,
        ])

        # self.xs = np.arange(x_bounds[0], x_bounds[1] + pixel_width, pixel_width)
        # self.ys = np.arange(y_bounds[1], y_bounds[0] - pixel_height, -pixel_height)

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.crs = crs

    @classmethod
    def from_referenced_images(cls, filename, reffed_images, crs):

        # Bounds
        x_bounds, y_bounds, pixel_width, pixel_height = get_containing_bounds(
            reffed_images,
            crs
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

    def bounds_to_offset(self, x_bounds, y_bounds):

        # Get offsets
        x_offset = x_bounds[0] - self.x_bounds[0]
        x_offset_count = int(np.round(x_offset / self.pixel_width))
        y_offset = self.y_bounds[1] - y_bounds[1]
        y_offset_count = int(np.round(y_offset / self.pixel_height))

        # Get width counts
        x_count = int(np.round((x_bounds[1] - x_bounds[0]) / self.pixel_width))
        y_count = int(np.round((y_bounds[1] - y_bounds[0]) / self.pixel_height))

        return x_offset_count, y_offset_count, x_count, y_count

    def get_img(self, x_bounds, y_bounds):

        x_offset_count, y_offset_count, x_count, y_count = self.bounds_to_offset(
            x_bounds,
            y_bounds,
        )

        img = self.dataset.ReadAsArray(xoff=x_offset_count, yoff=y_offset_count, xsize=x_count, ysize=y_count)
        return img.transpose(1, 2, 0)

    def get_referenced_image(self, x_bounds, y_bounds):

        img = self.get_img(x_bounds, y_bounds)

        reffed_image = observations.ReferencedImage(
            img,
            x_bounds,
            y_bounds,
            cart_crs_code='{}:{}'.format(*self.crs.to_authority()),
        )

        return reffed_image

    def save_img(self, img, x_bounds, y_bounds):

        x_offset_count, y_offset_count, x_count, y_count = self.bounds_to_offset(
            x_bounds,
            y_bounds,
        )

        img_to_save = img.transpose(2, 0, 1)
        self.dataset.WriteArray(img_to_save, xoff=x_offset_count, yoff=y_offset_count)

        self.dataset.FlushCache()
        
    def incorporate_referenced_image(self, src: observations.ReferencedImage):

        x_bounds, y_bounds = src.get_bounds(self.crs)
        dst_img = self.get_img(x_bounds, y_bounds)

        # Resize the image
        src_img_resized = cv2.resize(
            src.img_int[:, :, :3],
            (dst_img.shape[1], dst_img.shape[0])
        )

        # Blend
        is_empty = (dst_img.sum(axis=2) == 0)
        dst_img = np.array([
            np.where(is_empty, src_img_resized[:, :, j], dst_img[:, :, j])
            for j in range(3)
        ])
        dst_img = dst_img.transpose(1, 2, 0)

        self.save_img(dst_img, x_bounds, y_bounds)


def get_containing_bounds(reffed_images, crs):

    # Pixel size
    all_x_bounds = []
    all_y_bounds = []
    pixel_widths = []
    pixel_heights = []
    for i, reffed_image_i in enumerate(reffed_images):

        # Bounds
        x_bounds_i, y_bounds_i = reffed_image_i.get_bounds(crs)
        all_x_bounds.append(x_bounds_i)
        all_y_bounds.append(y_bounds_i)

        # Pixel properties
        pixel_width, pixel_height = reffed_image_i.get_pixel_widths()
        pixel_widths.append(pixel_width)
        pixel_heights.append(pixel_height)

    # Containing bounds
    all_x_bounds = np.array(all_x_bounds)
    all_y_bounds = np.array(all_y_bounds)
    x_bounds = [all_x_bounds[:, 0].min(), all_x_bounds[:, 1].max()]
    y_bounds = [all_y_bounds[:, 0].min(), all_y_bounds[:, 1].max()]

    # Use median pixel properties
    pixel_width = np.median(pixel_widths)
    pixel_height = np.median(pixel_heights)

    return x_bounds, y_bounds, pixel_width, pixel_height
