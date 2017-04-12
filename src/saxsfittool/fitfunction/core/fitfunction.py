import numpy as np


class FitFunction(object):
    """General class for univariate, single-valued fit functions, i.e:
        y=func(x, a1, a2, ...),

        where a1, a2, etc. are the model parameters, x is the single
        independent variable, y is the result. The function must be
        vectorized in x, and return a matrix of the same shape.

    Subclassing:
        1) override _function(): take care to return a matrix having
            the same shape as the first argument.

        2) if you want, override _jacobian(): it must return the
            Jacobian matrix of the function (row vector in the case
            of single-valued functions) containing the derivative
            of the function by each model argument.

        3) define the `name` attribute: just a short textual identifier

        4) `description` can be a longer, elaborate text, intended for
            tooltips

        5) `arguments` a list of model parameters. Each parameter has a
            tuple: (short name, description)
            
        6) `unfittable_parameters` parameters which are not intended to
            be fitted, but somehow influence the model function. E.g.
            number of points for numeric integration, etc. A list of
            tuples with (short name, description, minimum, maximum). These
            parameters are considered as integers.
    """
    name = ''

    description = ''

    arguments = []

    unfittable_parameters = []

    def __call__(self, x, *args):
        return self.function(x, *args)

    def function(self, x, *args):
        raise NotImplementedError

    def jacobian(self, x, *args):
        return None

    def initialize_arguments(self, x, y):
        return [1.0] * len(self.arguments)

    @classmethod
    def getsubclasses(cls):
        lis = []
        for sc in cls.__subclasses__():
            lis.append(sc)
            lis.extend(sc.getsubclasses())
        return [c for c in sorted(lis, key=lambda x: x.name) if c.name]

    def draw_representation(self, fig, x, *args):
        pass

class Linear(FitFunction):
    name = 'Linear'

    arguments = [('a', 'Slope'),
                 ('b', 'Y offset')]

    description = 'First-order polynomial'

    def function(self, x, a, b):
        return a * x + b

    def jacobian(self, x, a, b):
        return np.array([a, 1.0])

    def initialize_arguments(self, x, y):
        return np.array([1.0, 1.0])





class Gaussian(FitFunction):
    name = 'Gaussian peak'

    arguments = [('amplitude', 'Amplitude'),
                 ('center', 'Position'),
                 ('sigma', 'Half width at half maximum'),
                 ('offset', 'Constant offset')]

    description = 'Gaussian peak function'

    def function(self, x, amplitude, center, sigma, offset):
        return amplitude / (2 * np.pi * sigma ** 2) ** 0.5 * np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) + offset

