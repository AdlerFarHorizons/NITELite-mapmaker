'''This module handles georeferencing.
'''

from typing import Tuple

import numpy as np


class GeoReferencer:

    def resample(
        self,
        points: list[np.ndarray, np.ndarray],
        img: np.ndarray,
        res: float = None,
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

        # Calculate the weights (flux-conserved)
        xs, ys = points
        da = np.linalg.norm(np.cross(
            [xs[0, 1] - xs[0, 0], ys[0, 1] - ys[0, 0]],
            [xs[1, 0] - xs[0, 0], ys[1, 0] - ys[0, 0]],
        ))
        weights = img.flatten() * da

        # Setup output image
        if res is None:
            res = np.sqrt(da)
        x_bins = np.arange(xs.min() - res / 2., xs.max() + res / 2., res)
        y_bins = np.arange(ys.min() - res / 2., ys.max() + res / 2., res)

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

        # Get the coordinates out
        xs_resampled = 0.5 * (x_bins[:-1] + x_bins[1:])
        ys_resampled = 0.5 * (y_bins[:-1] + y_bins[1:])

        return (xs_resampled, ys_resampled), img_resampled
