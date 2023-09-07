'''Main mapmaker file.
'''

from . import observations


class Mapmaker:

    def __init__(self, **observation_kwargs):

        self.flight = observations.Flight(**observation_kwargs)

