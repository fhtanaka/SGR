import itertools as it
import numpy as np

def calc_layer(*coords):
    coord_arr = []
    for i in coords[:-1]:
        aux = np.linspace(-1.0, 1.0, i) if (i > 1) else [0.0]
        coord_arr.append(aux)
    last_coord = [coords[-1]]
    return tuple(it.product(*coord_arr, last_coord))

"""
The substrate.
"""

class Substrate(object):
    """
    Represents a substrate: Input coordinates, output coordinates, hidden coordinates and a resolution defaulting to 10.0.
    """

    def __init__(self, shape, res=10.0):
        self.res = res
        self.dimensions = len(shape[0])

        layers = [calc_layer(*l) for l in shape]

        self.input_coordinates = layers[0]
        self.output_coordinates  = layers[-1]
        self.hidden_coordinates = [[*l] for l in layers[1:-1]]
