'''Main mapmaker file.
'''
from typing import Union

import pyproj

from . import observations, data_viewer


class Mapmaker:

    def __init__(
        self,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        **observation_kwargs
    ):

        if isinstance(crs, str):
            crs = pyproj.CRS(crs)
        self.crs = crs

        # Data
        self.flight = observations.Flight(
            cart_crs_code=self.crs,
            **observation_kwargs
        )

        self.data_viewer = data_viewer.DataViewer(self.flight)

    def prep(self, metadata_fp: str = None, *args, **kwargs):
        '''Prepare to mapmake by loading metadata and/or prepping prep data
        other data for loading. We won't actually load the
        images because that can be >50 GB per camera per flight.
        '''

        self.flight.prep_metadata(metadata_fp, *args, **kwargs)

    def fit(
        self,
        fps: list[str],
        search_coords: list[tuple[float, float]],
        search_radii: list[float],
    ):

        # Make the mosaic
        pass
