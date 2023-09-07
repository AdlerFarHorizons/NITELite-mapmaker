'''Main mapmaker file.
'''

from . import observations, data_handler, data_viewer

class Mapmaker:

    def __init__(self, **observation_kwargs):

        # Main data
        self.flight = observations.Flight(**observation_kwargs)

        self.data_handler = data_handler.DataHandler(self.flight)
        self.data_viewer = data_viewer.DataViewer(self.flight)
