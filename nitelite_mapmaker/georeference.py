'''This module handles georeferencing.
'''

from typing import Tuple

import numpy as np


class ImageTransformer:
    '''Transforms images as needed to accompany geo-referencing.
    '''

    def __init__(self, img):

        self.img = img

    def resample(
        self,
        points: list[np.ndarray, np.ndarray] = None,
        bins: list[np.ndarray, np.ndarray] = None,
        preserve_flux: bool = True,
        return_counts: bool = False,
    ) -> Tuple[list[np.ndarray, np.ndarray], np.ndarray]:
        '''
        Perform a nearest-grid-point resample to rotate the array axes
        to be parallel to the coordinate system axes.

        GDAL (and other libraries, e.g. satpy) have much more thorough
        implementations that should be used for the final product.

        Args:
            points:
                x and y values for the pixels pre-resampling.
                Currently assumed that the grid is rectilinear. Adapting the
                code to work with non-rectilinear points may require doing an
                average of the cross products between neighboring pixels
                (to get the area) or using a triangulated irregular network.
            bins:
                x and y bins the array will be sampled onto.
                Defaults to slightly fewer pixels than the current image has.
                This does a nice job of preserving the image appearance,
                but introduces aliasing issues.
            preserve_flux:
                If True, deposit onto the new grid the value multiplied by
                the area, and return the deposited values scaled by the new
                pixel areas.
            return_counts:
                If True, also return an array indicating how many pixels
                were deposited onto each of the resampled values.
                This is essential for e.g. mosaic averaging.

        Returns:
            points_resampled: x and y values for the resampled pixels.
            img_resampled: Resampled image.
        '''

        if points is None:
            points = self.points
        xs, ys = points

        # Setup output image
        if bins is None:

            # This is an alternative that uses the nyquist frequency
            # dx = np.min(np.abs(np.diff(xs, axis=1))) / 2.
            # dy = np.min(np.abs(np.diff(ys, axis=0))) / 2.
            # x_bins = np.arange(
            #     xs.min() - dx / 2.,
            #     xs.max() + dx,
            #     dx
            # )
            # y_bins = np.arange(
            #     ys.min() - dy / 2.,
            #     ys.max() + dy,
            #     dy
            # )

            dx = np.mean(np.abs(np.diff(xs, axis=1)))
            dy = np.mean(np.abs(np.diff(xs, axis=1)))
            x_bins = np.linspace(
                xs.min() - dx / 2.,
                xs.max() + dx,
                int(np.floor(self.img.shape[1] * 0.99)),
            )
            y_bins = np.linspace(
                ys.min() - dy / 2.,
                ys.max() + dy,
                int(np.floor(self.img.shape[0] * 0.99)),
            )
            xs_resampled = 0.5 * (x_bins[:-1] + x_bins[1:])
            ys_resampled = 0.5 * (y_bins[:-1] + y_bins[1:])
        else:
            x_bins, y_bins = bins
            xs_resampled = 0.5 * (x_bins[:-1] + x_bins[1:])
            ys_resampled = 0.5 * (y_bins[:-1] + y_bins[1:])

        def resample_fn(img):
            '''Actual function for the resampling.
            Currently just a histogram binning, i.e. nearest grid point.
            '''
            img_resampled, _, _ = np.histogram2d(
                xs.flatten(),
                ys.flatten(),
                bins=[x_bins, y_bins],
                weights=img.flatten()
            )

            # Transpose, because histogram order is weird
            img_resampled = img_resampled.transpose()

            return img_resampled

        # No bands case.
        if len(self.img.shape) == 2:
            img_resampled = resample_fn(self.img)
        # Color image case.
        elif len(self.img.shape) == 3:
            img_resampled = np.array([
                resample_fn(self.img[:, :, i])
                for i in range(self.img.shape[2])
            ])
            img_resampled = img_resampled.transpose(1, 2, 0)
        else:
            raise ValueError(
                'Unexpected shape for self.img. Resample works with 2D images'
                ' or 2D images with multiple bands (3 dimensions).'
            )

        # Normalize bins
        if preserve_flux:
            da = np.linalg.norm(np.cross(
                [xs[0, 1] - xs[0, 0], ys[0, 1] - ys[0, 0]],
                [xs[1, 0] - xs[0, 0], ys[1, 0] - ys[0, 0]],
            ))
            da_resampled = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0])
            img_resampled *= da / da_resampled

        self.points_resampled = (xs_resampled, ys_resampled)
        self.bins = (x_bins, y_bins)
        self.img_resampled = img_resampled

        if return_counts:
            self.counts_resampled = resample_fn(
                np.ones(self.img.shape[:2])
            )
            return (
                self.points_resampled,
                self.img_resampled,
                self.counts_resampled
            )
        else:
            return self.points_resampled, self.img_resampled
