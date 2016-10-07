import numpy as np
from .c_ellipsoid import F2EllipsoidalShell

class FitFunction(object):
    """General class for a fit function"""
    name = ''

    description = ''

    arguments = []

    def __init__(self):
        self.fixed_argumens = []
        pass

    def __call__(self, x, *args, **kwargs):
        return self.function(x, *args, **kwargs)

    def function(self, x, *args, **kwargs):
        raise NotImplementedError

    def jacobian(self, x, *args, **kwargs):
        return None

    def initialize_arguments(self, x, y):
        return [1.0]*len(self.arguments)

    @classmethod
    def getsubclasses(cls):
        lis=[]
        for sc in cls.__subclasses__():
            print('Appending {}'.format(sc.name))
            lis.append(sc)
            lis.extend(sc.getsubclasses())
        print('Returning {}'.format(lis))
        return sorted(lis, key=lambda x:x.name)



class Linear(FitFunction):
    name = 'Linear'

    arguments = [('a', 'Slope'),
                 ('b', 'Y offset')]

    description = 'First-order polynomial'

    def function(self, x, a, b):
        return a*x+b

    def jacobian(self, x, a, b):
        return np.array([a, 1.0])

    def initialize_arguments(self, x, y):
        return np.array([1.0,1.0])

class F2CoreShellEllipsoid(FitFunction):
    name = 'Rotational core-shell ellipsoid'

    arguments = [('eta_core', 'SLD of the core'),
                 ('eta_shell', 'SLD of the shell'),
                 ('eta_solvent', 'SLD of the solvent'),
                 ('a', 'Principal semi-axis of the core'),
                 ('b', 'Equatorial semi-axis of the core'),
                 ('t', 'Shell thickness')]

    description = "Scattering intensity of a rotational core-shell ellipsoid"

    def function(self, x, eta_core, eta_shell, eta_solvent, a, b, t):
        return F2EllipsoidalShell(x, eta_core, eta_shell, eta_solvent, a, b, t)
