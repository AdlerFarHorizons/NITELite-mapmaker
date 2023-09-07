'''This module contains code for viewing data.
'''

import numpy as np

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt

from .observations import Flight


class DataViewer:

    def __init__(self, flight: Flight):

        self.flight = flight

    def plot_img(self, img: np.ndarray, ax: matplotlib.axes.Axes = None):
        '''Quick image plot with decent defaults.
        '''

        if ax is None:
            fig = plt.figure(figsize=np.array(self.flight.img_shape) / 60.)
            ax = plt.gca()

        ax.imshow(img)
        ax.set_aspect('equal')
