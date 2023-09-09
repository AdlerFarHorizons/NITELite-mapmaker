'''Main mapmaker file.
'''

from . import observations, data_viewer


class Mapmaker:

    def __init__(self, **observation_kwargs):

        # Main data
        self.flight = observations.Flight(**observation_kwargs)

        self.data_viewer = data_viewer.DataViewer(self.flight)

    def load(self, metadata_fp: str = None):
        '''Load data or prep data for loading. We won't actually load the
        images because that can be >50 GB per camera per flight.
        '''

        self.flight.load_metadata(metadata_fp)
