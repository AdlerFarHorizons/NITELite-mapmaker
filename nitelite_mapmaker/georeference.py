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
        points_resampled: list[np.ndarray, np.ndarray] = None
    ) -> Tuple[list[np.ndarray, np.ndarray], np.ndarray]:
        '''
        Args:
            points: x and y values for the pixels. Currently assumed that
                the grid is rectilinear. Adapting the code to work with
                non-rectilinear points may require doing an average of the
                cross products between neighboring pixels (to get the area)
                or using a triangulated irregular network.
            img: The array values themselves
            res: 1D resolution of the output. Currently defaults to the
                geometric mean of the input image x and y resoluion.

        Returns:
            points_resampled: x and y values for the resampled pixels.
            img_resampled: Resampled image.
        '''
        if points is None:
            xs, ys = self.points

        # Setup output image
        if points_resampled is None:
            dx = np.max(np.abs(np.diff(xs)))
            dy = np.max(np.abs(np.diff(ys)))
            x_bins = np.arange(
                xs.min() - dx / 2.,
                xs.max() + 1.1 * dx / 2.,
                dx
            )
            y_bins = np.arange(
                ys.min() - dy / 2.,
                ys.max() + 1.1 * dy / 2.,
                dy
            )
            xs_resampled = 0.5 * (x_bins[:-1] + x_bins[1:])
            ys_resampled = 0.5 * (y_bins[:-1] + y_bins[1:])
        else:
            xs_resampled, ys_resampled = points_resampled

            dx = xs_resampled[1] - xs_resampled[0]
            x_bins = np.empty(xs_resampled.size + 1)
            x_bins[:-1] = xs_resampled - dx / 2.
            x_bins[-1] = xs_resampled[-1] + dx / 2.

            dy = ys_resampled[1] - ys_resampled[0]
            y_bins = np.empty(ys_resampled.size + 1)
            y_bins[:-1] = ys_resampled - dy / 2.
            y_bins[-1] = ys_resampled[-1] + dy / 2.

        # Calculate the weights (flux-conserved)
        da = np.linalg.norm(np.cross(
            [xs[0, 1] - xs[0, 0], ys[0, 1] - ys[0, 0]],
            [xs[1, 0] - xs[0, 0], ys[1, 0] - ys[0, 0]],
        ))
        weights = self.img.flatten() * da

        # Resample (currently just binning w/ a histogram)
        img_resampled, _, _ = np.histogram2d(
            xs.flatten(),
            ys.flatten(),
            bins=[x_bins, y_bins],
            weights=weights
        )

        # Normalize bins
        da_resampled = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0])
        img_resampled /= da_resampled

        self.points_resampled = (xs_resampled, ys_resampled)
        self.img_resampled = img_resampled

        return self.points_resampled, self.img_resampled
