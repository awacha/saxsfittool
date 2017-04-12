import numpy as np

from .c_spheredistrib import F2GaussianSphereDistribution
from ..core import FitFunction


class GaussianSphereDistribution(FitFunction):
    name = "Gaussian sphere distribution (intensity weighted)"

    description = "Intensity weighted gaussian radius distribution of solid spheres"

    arguments = [('factor', 'Scaling factor'),
                 ('background', 'Constant background'),
                 ('r0', 'Mean radius'),
                 ('dr', 'HWHM radius'),
                 ]

    def initialize_arguments(self, x, y):
        return ((1, 0, 40, 1), (0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf))

    def function(self, x, factor, background, r0, dr):
        return factor * F2GaussianSphereDistribution(x, r0, dr, weighting='intensity') + background


class GaussianSphereDistribution_number(FitFunction):
    name = "Gaussian sphere distribution (number weighted)"

    description = "Number weighted gaussian radius distribution of solid spheres"

    arguments = [('factor', 'Scaling factor'),
                 ('background', 'Constant background'),
                 ('r0', 'Mean radius'),
                 ('dr', 'HWHM radius'),
                 ]

    def initialize_arguments(self, x, y):
        return ((1, 0, 40, 1), (0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf))

    def function(self, x, factor, background, r0, dr):
        return factor * F2GaussianSphereDistribution(x, r0, dr, weighting='number') + background


class GaussianSphereDistribution_mass(FitFunction):
    name = "Gaussian sphere distribution (mass weighted)"

    description = "Mass weighted gaussian radius distribution of solid spheres"

    arguments = [('factor', 'Scaling factor'),
                 ('background', 'Constant background'),
                 ('r0', 'Mean radius'),
                 ('dr', 'HWHM radius'),
                 ]

    def initialize_arguments(self, x, y):
        return ((1, 0, 40, 1), (0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf))

    def function(self, x, factor, background, r0, dr):
        return factor * F2GaussianSphereDistribution(x, r0, dr, weighting='mass') + background
