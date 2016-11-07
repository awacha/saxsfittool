import numpy as np
from matplotlib.figure import Figure

from .c_bilayer import F2FiveGaussSymmetricHeadBilayer
from .c_ellipsoid import F2EllipsoidalShell
from .c_spheredistrib import F2GaussianSphereDistribution


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

        5) `argument` a list of model parameters. Each parameter has a
            tuple: (short name, description)
    """
    name = ''

    description = ''

    arguments = []

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
        return sorted(lis, key=lambda x: x.name)

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

class F2CoreShellEllipsoidWithBackground(FitFunction):
    name = 'Rotational core-shell ellipsoid + constant background'

    arguments = [('eta_core', 'SLD of the core'),
                 ('eta_shell', 'SLD of the shell'),
                 ('eta_solvent', 'SLD of the solvent'),
                 ('a', 'Principal semi-axis of the core'),
                 ('b', 'Equatorial semi-axis of the core'),
                 ('t', 'Shell thickness'),
                 ('bg', 'Constant background')]

    description = "Scattering intensity of a rotational core-shell ellipsoid with additional constant background"

    def function(self, x, eta_core, eta_shell, eta_solvent, a, b, t, C):
        return F2EllipsoidalShell(x, eta_core, eta_shell, eta_solvent, a, b, t) + C


class Gaussian(FitFunction):
    name = 'Gaussian peak'

    arguments = [('amplitude', 'Amplitude'),
                 ('center', 'Position'),
                 ('sigma', 'Half width at half maximum'),
                 ('offset', 'Constant offset')]

    description = 'Gaussian peak function'

    def function(self, x, amplitude, center, sigma, offset):
        return amplitude / (2 * np.pi * sigma ** 2) ** 0.5 * np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) + offset


class FiveGaussianSymmetricHeadBilayer(FitFunction):
    name = 'Bilayer with guests'

    description = 'Symmetric head bilayer with inner and outer bilayers'

    arguments = [('factor', 'Scaling factor'),
                 ('background', 'Constant background level'),
                 ('meanradius', 'Mean radius of the tail region'),
                 ('deltaradius', 'Spread of the radii of the tail region'),
                 ('rho_guest_in', 'Relative SLD of the inner guest layer'),
                 ('rho_head', 'Relative SLD of the head groups'),
                 ('rho_guest_out', 'Relative SLD of the outer guest layer'),
                 ('delta_guest_in', '(Positive) distance of the inner guest layer from the tail region'),
                 ('delta_head', '(Positive) distance of the headgroup layer from the tail region'),
                 ('delta_guest_out', '(Positive) distance of the outer guest layer from the tail region'),
                 ('sigma_guest_in', '(Positive) HWHM of the inner guest layer'),
                 ('sigma_head', '(Positive) HWHM of the headgroup layer'),
                 ('sigma_tail', '(Positive) HWHM of the tail layer'),
                 ('sigma_guest_out', '(Positive) HWHM of the outer guest layer'),
                 ]

    def function(self, x, factor, bg, R, dR, rho_guest_in, rho_head, rho_guest_out, delta_guest_in,
                 delta_head, delta_guest_out, sigma_guest_in, sigma_head, sigma_tail, sigma_guest_out):
        return factor * F2FiveGaussSymmetricHeadBilayer(
            x, R, dR, 0.0, 0.0, rho_guest_in, rho_head, rho_guest_out, delta_guest_in,
            delta_head, delta_guest_out, sigma_guest_in, sigma_head,
            sigma_tail, sigma_guest_out, 0, 1) + bg

    def initialize_arguments(self, x, y):
        return (np.array([0.0000001, 0, 40, 10, 0.1, 1, 0.1, 10, 2.5, 10, 1,1,1,1]),
                np.array([0,         0,  0,  0, 0,   0, 0,   0,  0,   0,  0,0,0,0]),
                np.array([0.0001,    0.1,150,100,1,2,1,30,10,30,30,5,5,30]))

    @staticmethod
    def gaussian(r, rho, center, sigma):
        return rho / (2 * np.pi * sigma ** 2) * np.exp(-(r - center) ** 2 / (2 * sigma ** 2))

    def draw_representation(self, fig, x, factor, bg, R, dR, rho_guest_in, rho_head, rho_guest_out, delta_guest_in,
                            delta_head, delta_guest_out, sigma_guest_in, sigma_head, sigma_tail, sigma_guest_out):
        assert isinstance(fig, Figure)
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        hwhm_multiplier = 3
        rmin = min(R - delta_guest_in - hwhm_multiplier * sigma_guest_in,
                   R - delta_head - hwhm_multiplier * sigma_head,
                   R - hwhm_multiplier * sigma_tail)
        rmax = max(R + delta_guest_out + hwhm_multiplier * sigma_guest_out,
                   R + delta_head + hwhm_multiplier * sigma_head,
                   R + hwhm_multiplier * sigma_tail)
        r = np.linspace(rmin, rmax, 1000)
        sld_tail = self.gaussian(r, -1, R, sigma_tail)
        sld_head = self.gaussian(r, rho_head, R - delta_head, sigma_head) + self.gaussian(r, rho_head, R + delta_head,
                                                                                          sigma_head)
        sld_guest_in = self.gaussian(r, rho_guest_in, R - delta_guest_in, sigma_guest_in)
        sld_guest_out = self.gaussian(r, rho_guest_out, R + delta_guest_out, sigma_guest_out)
        ax.plot(r, sld_tail, label='Lipid chain region')
        ax.plot(r, sld_head, label='Lipid headgroup region')
        ax.plot(r, sld_guest_in, label='Inner guest molecules')
        ax.plot(r, sld_guest_out, label='Outer guest molecules')
        ax.plot(r, sld_tail + sld_head + sld_guest_in + sld_guest_out, label='Total')
        ax.set_xlabel('Radial distance from liposome center (nm)')
        ax.set_ylabel('Relative scattering length density')
        ax.legend(loc='best')
        ax.grid(True, which='both')
        fig.canvas.draw()


class FiveGaussianSymmetricHeadBilayerMultilamellar(FitFunction):
    name = 'Bilayer with guests, multilamellarity'

    description = 'Oligolamellar symmetric head bilayer with inner and outer bilayers'

    arguments = [('factor', 'Scaling factor'),
                 ('background', 'Constant background level'),
                 ('meanradius', 'Mean radius of the tail region'),
                 ('deltaradius', 'Spread of the radii of the tail region'),
                 ('rho_guest_in', 'Relative SLD of the inner guest layer'),
                 ('rho_head', 'Relative SLD of the head groups'),
                 ('rho_guest_out', 'Relative SLD of the outer guest layer'),
                 ('delta_guest_in', '(Positive) distance of the inner guest layer from the tail region'),
                 ('delta_head', '(Positive) distance of the headgroup layer from the tail region'),
                 ('delta_guest_out', '(Positive) distance of the outer guest layer from the tail region'),
                 ('sigma_guest_in', '(Positive) HWHM of the inner guest layer'),
                 ('sigma_head', '(Positive) HWHM of the headgroup layer'),
                 ('sigma_tail', '(Positive) HWHM of the tail layer'),
                 ('sigma_guest_out', '(Positive) HWHM of the outer guest layer'),
                 ('d', 'Expectance of the periodicity'),
                 ('dd', '(Positive) HWHM of the periodicity'),
                 ('x_oligolam', 'Ratio of oligolamellarity (0<=x<=1)'),
                 ('N_bilayers', 'Number of bilayers (integer, NOT FITTABLE)')
                 ]

    def function(self, x, factor, bg, R, dR, rho_guest_in, rho_head, rho_guest_out, delta_guest_in,
                 delta_head, delta_guest_out, sigma_guest_in, sigma_head, sigma_tail, sigma_guest_out, d, dd,
                 x_oligolam, Nbilayers):
        return factor * F2FiveGaussSymmetricHeadBilayer(
            x, R, dR, d, dd, rho_guest_in, rho_head, rho_guest_out, delta_guest_in,
            delta_head, delta_guest_out, sigma_guest_in, sigma_head,
            sigma_tail, sigma_guest_out, x_oligolam, int(Nbilayers)) + bg

    def initialize_arguments(self, x, y):
        return (np.array([0.0000001, 0, 40, 10, 0.1, 1, 0.1, 10, 2.5, 10, 1, 1, 1, 1, 6.4, 0.3, 0, 2]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0.0001, 0.1, 150, 100, 1, 2, 1, 30, 10, 30, 30, 5, 5, 30, 20, 10, 1, 8]))

    @staticmethod
    def gaussian(r, rho, center, sigma):
        return rho / (2 * np.pi * sigma ** 2) * np.exp(-(r - center) ** 2 / (2 * sigma ** 2))

    def draw_representation(self, fig, x, factor, bg, R, dR, rho_guest_in, rho_head, rho_guest_out, delta_guest_in,
                            delta_head, delta_guest_out, sigma_guest_in, sigma_head, sigma_tail, sigma_guest_out, d, dd,
                            x_oligolam, Nbilayers):
        assert isinstance(fig, Figure)
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        hwhm_multiplier = 3
        rmin = min(R - delta_guest_in - hwhm_multiplier * sigma_guest_in,
                   R - delta_head - hwhm_multiplier * sigma_head,
                   R - hwhm_multiplier * sigma_tail)
        rmax = max(R + delta_guest_out + hwhm_multiplier * sigma_guest_out,
                   R + delta_head + hwhm_multiplier * sigma_head,
                   R + hwhm_multiplier * sigma_tail)
        r = np.linspace(rmin, rmax, 1000)
        sld_tail = self.gaussian(r, -1, R, sigma_tail)
        sld_head = self.gaussian(r, rho_head, R - delta_head, sigma_head) + self.gaussian(r, rho_head, R + delta_head,
                                                                                          sigma_head)
        sld_guest_in = self.gaussian(r, rho_guest_in, R - delta_guest_in, sigma_guest_in)
        sld_guest_out = self.gaussian(r, rho_guest_out, R + delta_guest_out, sigma_guest_out)
        ax.plot(r, sld_tail, label='Lipid chain region')
        ax.plot(r, sld_head, label='Lipid headgroup region')
        ax.plot(r, sld_guest_in, label='Inner guest molecules')
        ax.plot(r, sld_guest_out, label='Outer guest molecules')
        ax.plot(r, sld_tail + sld_head + sld_guest_in + sld_guest_out, label='Total')
        ax.set_xlabel('Radial distance from liposome center (nm)')
        ax.set_ylabel('Relative scattering length density')
        ax.legend(loc='best')
        ax.grid(True, which='both')
        fig.canvas.draw()


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
