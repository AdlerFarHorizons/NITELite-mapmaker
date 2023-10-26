import numpy as np
import pandas as pd
import scipy
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

import copy
import glob
import os

import cv2
from osgeo import gdal, gdal_array
import pyproj

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from typing import Tuple


class Image:

    def __init__(self, img):
        if np.issubdtype(img.dtype, np.floating):
            self.img = img
        elif np.issubdtype(img.dtype, np.integer):
            self.img_int = img
            self.img = (img / 255).astype(np.float32)

    @property
    def img(self):
        '''Image property for quick access. For the base class Image
        it's very simple, but will be overwritten by other classes.
        '''
        return self._img

    @img.setter
    def img(self, value):
        self._img = value

    @property
    def img_int(self) -> np.ndarray[int]:
        if not hasattr(self, '_img_int'):
            self._img_int = self.get_img_int_from_img()
        return self._img_int

    @img_int.setter
    def img_int(self, value):
        self._img_int = value

    @property
    def img_shape(self):
        return self._img.shape[:2]

    @property
    def semitransparent_img(self) -> np.ndarray[float]:
        if not hasattr(self, '_semitransparent_img'):
            self._semitransparent_img = self.get_semitransparent_img()
        return self._semitransparent_img

    @property
    def kp(self):
        if not hasattr(self, '_kp'):
            self.get_features()
        return self._kp

    @property
    def des(self):
        if not hasattr(self, '_des'):
            self.get_features()
        return self._des

    def get_img_int_from_img(self) -> np.ndarray[int]:
        '''

        TODO: State (and assess) general principle--
            will use default options for image retrieval.
            For more fine-grained control call get_img
            first, instead of passing in additional arguments.
        '''

        img_int = (self.img * 255).astype(np.uint8)

        return img_int

    def get_nonzero_mask(self) -> np.ndarray[bool]:

        return self.img_int.sum(axis=2) > 0

    def get_semitransparent_img(self) -> np.ndarray[float]:

        semitransparent_img = np.zeros(
            shape=(self.img_shape[0], self.img_shape[1], 4)
        )
        semitransparent_img[:, :, :3] = self.img
        semitransparent_img[:, :, 3] = self.get_nonzero_mask().astype(float)

        return semitransparent_img

    def get_features(self):

        orb = cv2.ORB_create()

        self._kp, self._des = orb.detectAndCompute(self.img_int, None)

        return self._kp, self._des

    def get_pixel_coordinates(self):

        pxs = np.arange(self.img_shape[1])
        pys = np.arange(self.img_shape[0])

        return pxs, pys

    def plot_kp(
        self,
        ax=None,
        kp=None,
        colors=None,
        crs_transform=None,
        cmap='viridis',
        vmin=None,
        vmax=None,
        *args,
        **kwargs
    ):

        if ax is None:
            fig = plt.figure(figsize=np.array(self.img_shape) / 60.)
            ax = plt.gca()

        # KP details retrieval
        if kp is None:
            kp = self.kp
        kp_xs, kp_ys = np.array([_.pt for _ in kp]).transpose()
        if colors is None:
            colors = np.array([_.response for _ in kp])

        # Transform to appropriate coordinate system
        if crs_transform is not None:
            kp_xs, kp_ys = crs_transform(kp_xs, kp_ys)

        # Colormap
        cmap = sns.color_palette(cmap, as_cmap=True)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        # Argument update
        used_kwargs = {
            'c': 'none',
            'marker': 'o',
            's': 150,
            'linewidth': 2,
        }
        used_kwargs.update(kwargs)

        # Plot itself
        s = ax.scatter(
            kp_xs,
            kp_ys,
            edgecolors=cmap(norm(colors)),
            *args,
            **used_kwargs
        )

        return s

    def show(self, ax=None, img='img', *args, **kwargs):
        '''
            NOTE: This will not be consistent with imshow, because with imshow
        the y-axis increases downwards, consistent with old image
        processing schemes. Instead this is consistent with transposing and
        positively scaling the image to cartesian coordinates.

        Args:
        Kwargs:
        Returns:
        '''

        if ax is None:
            fig = plt.figure(figsize=np.array(self.img_shape) / 60.)
            ax = plt.gca()

        pxs, pys = self.get_pixel_coordinates()

        ax.pcolormesh(
            pxs,
            pys,
            getattr(self, img),
            *args,
            **kwargs
        )

        ax.set_aspect('equal')


# class Dataset:
#     '''Wrapper for GDAL Dataset.
#     '''


class ReferencedImage(Image):

    def __init__(
        self,
        img,
        x_bounds,
        y_bounds,
        cart_crs_code: str = 'EPSG:3857',
        latlon_crs_code: str = 'EPSG:4326',
    ):

        super().__init__(img)

        # Make a gdal dataset
        gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(
            self.img_int.dtype
        )
        driver = gdal.GetDriverByName('MEM')
        dataset = driver.Create(
            '',
            self.img_int.shape[1],
            self.img_int.shape[0],
            self.img_int.shape[2],
            gdal_dtype
        )

        # Write data
        for i in range(self.img_int.shape[2]):
            dataset.GetRasterBand(i + 1).WriteArray(self.img_int[:, :, i])

        # Establish CRS and conversions
        self.cart_crs_code = cart_crs_code
        self.latlon_crs_code = latlon_crs_code
        self.cart_crs = pyproj.CRS(cart_crs_code)
        self.latlon_crs = pyproj.CRS(latlon_crs_code)
        self.cart_to_latlon = pyproj.Transformer.from_crs(
            self.cart_crs,
            self.latlon_crs
        )
        self.latlon_to_cart = pyproj.Transformer.from_crs(
            self.latlon_crs,
            self.cart_crs
        )
        dataset.SetProjection(self.cart_crs.to_wkt())

        # Set geotransform
        dx = (x_bounds[1] - x_bounds[0]) / self.img_shape[1]
        dy = (y_bounds[1] - y_bounds[0]) / self.img_shape[0]
        geotransform = (
            x_bounds[0],
            dx,
            0,
            y_bounds[1],
            0,
            -dy
        )
        dataset.SetGeoTransform(geotransform)

        self.dataset = dataset

    @property
    def latlon_bounds(self):
        if not hasattr(self, '_latlon_bounds'):
            self._latlon_bounds = self.get_bounds(self.latlon_crs)
        return self._latlon_bounds

    @property
    def cart_bounds(self):
        if not hasattr(self, '_cart_bounds'):
            self._cart_bounds = self.get_bounds(self.cart_crs)
        return self._cart_bounds

    @property
    def img_shape(self):
        if hasattr(self, '_img'):
            return self._img.shape[:2]
        else:
            return (self.dataset.RasterYSize, self.dataset.RasterXSize)

    def get_bounds(self, crs: pyproj.CRS) -> Tuple[np.ndarray, np.ndarray]:
        '''Get image bounds in a given coordinate system.

        Args:
            crs: Desired coordinate system.

        Returns:
            x_bounds: x_min, x_max of the image in the target coordinate system
            y_bounds: y_min, y_max of the image in the target coordinate system
        '''

        # Get the coordinates
        x_min, px_width, x_rot, y_max, y_rot, px_height = \
            self.dataset.GetGeoTransform()
        x_max = x_min + px_width * self.dataset.RasterXSize
        y_min = y_max + px_height * self.dataset.RasterYSize

        # Convert to desired crs
        dataset_crs = pyproj.CRS(self.dataset.GetProjection())
        dataset_to_desired = pyproj.Transformer.from_crs(
            dataset_crs,
            crs,
            always_xy=True
        )
        x_bounds, y_bounds = dataset_to_desired.transform(
            [x_min, x_max],
            [y_min, y_max]
        )

        return x_bounds, y_bounds

    def get_cart_coordinates(self):

        x_bounds, y_bounds = self.cart_bounds

        xs = np.linspace(x_bounds[0], x_bounds[1], self.img_shape[1])
        ys = np.linspace(y_bounds[1], y_bounds[0], self.img_shape[0])

        return xs, ys

    def get_pixel_widths(self):
        xs, ys = self.get_cart_coordinates()
        return np.abs(xs[1] - xs[0]), np.abs(ys[1] - ys[0])

    def get_pixel_coordinates(self):

        pxs = np.arange(self.dataset.RasterXSize)
        pys = np.arange(self.dataset.RasterYSize)

        return pxs, pys

    def convert_pixel_to_cart(self, pxs, pys):

        (x_min, x_max), (y_min, y_max) = self.cart_bounds

        x_scaling = (x_max - x_min) / (self.dataset.RasterXSize - 1)
        y_scaling = (y_min - y_max) / (self.dataset.RasterYSize - 1)

        xs = x_scaling * pxs + x_min
        ys = y_scaling * pys + y_max

        return xs, ys

    def convert_cart_to_pixel(self, xs, ys):

        (x_min, x_max), (y_min, y_max) = self.cart_bounds

        x_scaling = (self.dataset.RasterXSize - 1) / (x_max - x_min)
        y_scaling = (self.dataset.RasterYSize - 1) / (y_min - y_max)

        pxs = (xs - x_min) * x_scaling
        pys = (ys - y_max) * y_scaling

        return pxs, pys

    def plot_bounds(self, ax, *args, **kwargs):

        used_kwargs = {
            'linewidth': 3,
            'facecolor': 'none',
            'edgecolor': '#dd8452',
        }
        used_kwargs.update(kwargs)

        x_bounds, y_bounds = self.cart_bounds
        x_min = x_bounds[0]
        y_min = y_bounds[0]
        width = x_bounds[1] - x_bounds[0]
        height = y_bounds[1] - y_bounds[0]

        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            *args,
            **used_kwargs
        )
        ax.add_patch(rect)

    def plot_kp(self, ax=None, crs_transform='cartesian', *args, **kwargs):

        if crs_transform == 'cartesian':
            crs_transform = self.convert_pixel_to_cart

        return super().plot_kp(
            ax=ax,
            crs_transform=crs_transform,
            *args,
            **kwargs
        )

    def show(self, ax=None, img='img', crs='pixel', *args, **kwargs):
        '''
        TODO: Make this more consistent with naming of other functions?
        '''

        # Use existing functionality
        if crs == 'pixel':
            return super().show(ax=ax, img=img, *args, **kwargs)

        if ax is None:
            fig = plt.figure(figsize=np.array(self.img_shape) / 60.)
            ax = plt.gca()

        xs, ys = self.get_cart_coordinates()

        ax.pcolormesh(
            xs,
            ys,
            getattr(self, img),
            *args,
            **kwargs
        )

        ax.set_aspect('equal')

    def add_to_folium_map(
        self,
        m,
        img: str = 'semitransparent_img',
        label: str = 'referenced',
        include_corner_markers: bool = False
    ):
        '''Add to a folium map.
        
        Args:
            m (folium map): The map to add to.
        '''

        # Let's keep this as an optional import for now.
        import folium

        lon_bounds, lat_bounds = self.latlon_bounds
        bounds = [
            [lat_bounds[0], lon_bounds[0]],
            [lat_bounds[1], lon_bounds[1]]
        ]
        img_arr = getattr(self, img)

        folium.raster_layers.ImageOverlay(
            img_arr,
            bounds=bounds,
            name=label,
        ).add_to(m)

        # Markers for the corners so we can understand how the image pixels
        # get flipped around
        if include_corner_markers:
            bounds_group = folium.FeatureGroup(name=f'{label} bounds')
            minmax_labels = ['min', 'max']
            for ii in range(2):
                for jj in range(2):
                    hover_text = (
                        f'(x_{minmax_labels[jj]}, '
                        f'y_{minmax_labels[ii]})'
                    )
                    folium.Marker(
                        [lat_bounds[ii], lon_bounds[jj]],
                        popup=hover_text,
                        icon=folium.Icon(
                            color='gray',
                            # TODO: Incorporate colors
                            # icon_color=palette.as_hex()[jj * 2 + ii]
                        ),
                    ).add_to(bounds_group)
            bounds_group.add_to(m)
