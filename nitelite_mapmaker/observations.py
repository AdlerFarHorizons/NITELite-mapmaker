'''This module handles observations, e.g. NITELite flights.
'''


class Flight:

    def __init__(self, image_dir, metadata_fp):

        self.image_dir = image_dir
        self.metadata_fp = metadata_fp
