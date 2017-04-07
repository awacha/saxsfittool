import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from .c_bilayer import F2FiveGaussSymmetricHeadBilayer
from .c_ellipsoid import F2EllipsoidalShell, F2AsymmetricEllipsoidalShell
from .c_ellipsoid2 import AsymmetricEllipsoidalShell
from .c_spheredistrib import F2GaussianSphereDistribution, F2GaussianCoreShellSphereDistribution


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

class F2AsymmetricCoreShellEllipsoid(FitFunction):
    name = 'Rotational core-shell ellipsoid with asymmetric shell'

    arguments = [('eta_core', 'SLD of the core'),
                 ('eta_shell', 'SLD of the shell'),
                 ('eta_solvent', 'SLD of the solvent'),
                 ('a', 'Principal semi-axis of the core'),
                 ('b', 'Equatorial semi-axis of the core'),
                 ('ta', 'Shell thickness (along the principal axis)'),
                 ('tb', 'Shell thickness (along the equatorial axes)'),
                 ]

    description = "Scattering intensity of an asymmetric rotational core-shell ellipsoid"

    def function(self, x, eta_core, eta_shell, eta_solvent, a, b, ta, tb):
        return F2AsymmetricEllipsoidalShell(x, eta_core, eta_shell, eta_solvent, a, b, ta, tb)


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

class F2AsymmetricCoreShellEllipsoidWithBackground(FitFunction):
    name = 'Rotational core-shell ellipsoid with asymmetric shell + constant background'

    arguments = [('eta_core', 'SLD of the core'),
                 ('eta_shell', 'SLD of the shell'),
                 ('eta_solvent', 'SLD of the solvent'),
                 ('a', 'Principal semi-axis of the core'),
                 ('b', 'Equatorial semi-axis of the core'),
                 ('ta', 'Shell thickness (along the principal axis)'),
                 ('tb', 'Shell thickness (along the equatorial axes)'),
                 ('bg', 'Constant background')]

    description = "Scattering intensity of an asymmetric rotational core-shell ellipsoid with additional constant background"

    def function(self, x, eta_core, eta_shell, eta_solvent, a, b, ta, tb, C):
        return F2AsymmetricEllipsoidalShell(x, eta_core, eta_shell, eta_solvent, a, b, ta, tb) + C

class F2AsymmetricCoreShellEllipsoidWithBackgroundI0RgA(FitFunction):
    name = 'Rotational core-shell ellipsoid + background, I0 and Rg parameters, branch A'

    arguments = [('I0', 'Intensity extrapolated to zero'),
                 ('Rg', 'Radius of gyration'),
                 ('a', 'Principal semi-axis of the core'),
                 ('b', 'Equatorial semi-axis of the core'),
                 ('ta', 'Shell thickness (along the principal axis)'),
                 ('tb', 'Shell thickness (along the equatorial axes)'),
                 ('bg', 'Constant background')]

    description = "Scattering intensity of an asymmetric rotational core-shell ellipsoid with additional constant background, branch A"

    def function(self, x, I0, Rg, a, b, ta, tb, C):
        eta_solvent = 0
        btb=b+tb
        ata=a+ta
        btb2ata = btb**2*ata
        b2a = b**2*a
        rhocoredivrhoshell = (ata**3*btb**2+2*ata*btb**4-5*Rg**2*ata*btb**2-a**3*b**2-2*a*b**4+5*a*b**2*Rg**2)/(5*a*b**2*Rg**2-a**3*b**2-2*a*b**4)
        eta_shell = 3*I0**0.5/4/np.pi/(rhocoredivrhoshell*b2a+btb2ata-b2a)
        eta_core = rhocoredivrhoshell*eta_shell
        return AsymmetricEllipsoidalShell(x, eta_core, eta_shell, a, b, ta, tb) + C

    def draw_representation(self, fig, x, I0, Rg, a, b, ta, tb, C):
        eta_solvent = 0
        btb=b+tb
        ata=a+ta
        btb2ata = btb**2*ata
        b2a = b**2*a
        rhocoredivrhoshell = (ata**3*btb**2+2*ata*btb**4-5*Rg**2*ata*btb**2-a**3*b**2-2*a*b**4+5*a*b**2*Rg**2)/(5*a*b**2*Rg**2-a**3*b**2-2*a*b**4)
        eta_shell = 3*I0**0.5/4/np.pi/(rhocoredivrhoshell*b2a+btb2ata-b2a)
        eta_core = rhocoredivrhoshell*eta_shell
        fig.clear()
        ax=fig.add_subplot(1,1,1)
        assert isinstance(ax, Axes)
        p=Ellipse((0,0),2*(b+tb),2*(a+ta),color='yellow')
        ax.add_patch(p)
        p=Ellipse((0,0),2*(b),2*(a), color='green')
        ax.add_patch(p)
        ax.vlines(0,-a-ta,a+ta,linestyle='--',color='black')
        ax.autoscale_view(True, True, True)
        ax.text(0.05,0.95,
                '$a$: {}\n'
                '$b$: {}\n'
                '$t_a$: {}\n'
                '$t_b$: {}\n'
                '$\\rho_\mathrm{{core}}$: {}\n'
                '$\\rho_\mathrm{{shell}}$: {}\n'
                '$I_0$: {}\n'
                '$R_g$: {}\n'.format(a,b,ta,tb, eta_core, eta_shell, I0, Rg),
                transform=ax.transAxes,ha='left',va='top')
        fig.canvas.draw()

class F2AsymmetricCoreShellEllipsoidWithBackgroundI0RgB(FitFunction):
    name = 'Rotational core-shell ellipsoid + background, I0 and Rg parameters, branch B'

    arguments = [('I0', 'Intensity extrapolated to zero'),
                 ('Rg', 'Radius of gyration'),
                 ('a', 'Principal semi-axis of the core'),
                 ('b', 'Equatorial semi-axis of the core'),
                 ('ta', 'Shell thickness (along the principal axis)'),
                 ('tb', 'Shell thickness (along the equatorial axes)'),
                 ('bg', 'Constant background')]

    description = "Scattering intensity of an asymmetric rotational core-shell ellipsoid with additional constant background, branch B"

    def function(self, x, I0, Rg, a, b, ta, tb, C):
        eta_solvent = 0
        btb=b+tb
        ata=a+ta
        btb2ata = btb**2*ata
        b2a = b**2*a
        rhocoredivrhoshell = (ata**3*btb**2+2*ata*btb**4-5*Rg**2*ata*btb**2-a**3*b**2-2*a*b**4+5*a*b**2*Rg**2)/(5*a*b**2*Rg**2-a**3*b**2-2*a*b**4)
        eta_shell = -3*I0**0.5/4/np.pi/(rhocoredivrhoshell*b2a+btb2ata-b2a)
        eta_core = rhocoredivrhoshell*eta_shell
        return AsymmetricEllipsoidalShell(x, eta_core, eta_shell, a, b, ta, tb) + C

    def draw_representation(self, fig, x, I0, Rg, a, b, ta, tb, C):
        eta_solvent = 0
        btb=b+tb
        ata=a+ta
        btb2ata = btb**2*ata
        b2a = b**2*a
        rhocoredivrhoshell = (ata**3*btb**2+2*ata*btb**4-5*Rg**2*ata*btb**2-a**3*b**2-2*a*b**4+5*a*b**2*Rg**2)/(5*a*b**2*Rg**2-a**3*b**2-2*a*b**4)
        eta_shell = -3*I0**0.5/4/np.pi/(rhocoredivrhoshell*b2a+btb2ata-b2a)
        eta_core = rhocoredivrhoshell*eta_shell
        fig.clear()
        ax=fig.add_subplot(1,1,1)
        assert isinstance(ax, Axes)
        p=Ellipse((0,0),2*(b+tb),2*(a+ta),color='yellow')
        ax.add_patch(p)
        p=Ellipse((0,0),2*(b),2*(a), color='green')
        ax.add_patch(p)
        ax.vlines(0,-a-ta,a+ta,linestyle='--',color='black')
        ax.autoscale_view(True, True, True)
        ax.text(0.05,0.95,
                '$a$: {}\n'
                '$b$: {}\n'
                '$t_a$: {}\n'
                '$t_b$: {}\n'
                '$\\rho_\mathrm{{core}}$: {}\n'
                '$\\rho_\mathrm{{shell}}$: {}\n'
                '$I_0$: {}\n'
                '$R_g$: {}\n'.format(a,b,ta,tb, eta_core, eta_shell, I0, Rg),
                transform=ax.transAxes,ha='left',va='top')
        fig.canvas.draw()



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

class CoreShellSphereGaussianDistribution(FitFunction):
    name = "Spherical core-shell particles, Gaussian size distribution"

    description = "Intensity weighted distribution of spherical core-shell nanoparticles"

    arguments = [('factor', 'Scaling factor'),
                 ('background', 'Constant background'),
                 ('rcore', 'Mean core radius'),
                 ('sigmacore', 'HWHM of the core radius'),
                 ('tshell', 'Shell thickness'),
                 ('rhoshell_relative', 'SLD of the shell: the core is -1')]

    def function(self, x, factor, background, rcore, sigmacore, tshell, rhoshell_relative):
        return factor * F2GaussianCoreShellSphereDistribution(x, rcore, tshell, sigmacore, -1.0, rhoshell_relative) + background

class CoreShellSphereGaussianDistributionRgI0A(FitFunction):
    name = "Gaussian distribution of core-shell spheres, parametrized with Rg and I0, branch A"

    description = "Spherical core-shell nanoparticles, with Rg and I0, branch A"

    arguments = [('I0', 'Intensity extrapolated to zero'),
                 ('background', 'Constant background'),
                 ('Rg', 'Radius of gyration'),
                 ('rcore', 'Mean core radius'),
                 ('sigmacore', 'HWHM of the core radius'),
                 ('tshell', 'Shell thickness'),
                 ]

    def function(self, x, I0, background, Rg, rcore, sigmacore, tshell):
        R5=rcore**5
        R3=rcore**3
        Rpt3= (rcore+tshell)**3
        Rpt5 = (rcore+tshell)**5
        Rg253=5/3*Rg**2
        rhocoredivrhoshell=(Rpt5-R5-Rg253*(Rpt3-R3))/(Rg253*R3-R5)
        rhoshell = 3*I0**0.5/4/np.pi/(Rpt3-R3+R3*rhocoredivrhoshell)
        rhocore = rhoshell*rhocoredivrhoshell
        return F2GaussianCoreShellSphereDistribution(x, rcore, tshell, sigmacore, rhocore, rhoshell) + background

    def draw_representation(self, fig, x, I0, background, Rg, rcore, sigmacore, tshell):
        R5=rcore**5
        R3=rcore**3
        Rpt3= (rcore+tshell)**3
        Rpt5 = (rcore+tshell)**5
        Rg253=5/3*Rg**2
        fig.clear()
        ax=fig.add_subplot(1,1,1)
        assert isinstance(ax, Axes)
        p=Ellipse((0,0),2*(rcore+tshell),2*(rcore+tshell),color='yellow')
        ax.add_patch(p)
        p=Ellipse((0,0),2*(rcore),2*(rcore), color='green')
        ax.add_patch(p)
        rhocoredivrhoshell=(Rpt5-R5-Rg253*(Rpt3-R3))/(Rg253*R3-R5)
        rhoshell = 3*I0**0.5/4/np.pi/(Rpt3-R3+R3*rhocoredivrhoshell)
        rhocore = rhoshell*rhocoredivrhoshell
        ax.autoscale_view(True, True, True)
        ax.text(0.05,0.95,
                '$R_\mathrm{{core}}$: {}\n'
                '$T_\mathrm{{shell}}$: {}\n'
                '$\\rho_\mathrm{{core}}$: {}\n'
                '$\\rho_\mathrm{{shell}}$: {}\n'
                '$I_0$: {}\n'
                '$R_g$: {}\n'.format(rcore, tshell, rhocore, rhoshell, I0, Rg),
                transform=ax.transAxes,ha='left',va='top')
        fig.canvas.draw()


class CoreShellSphereGaussianDistributionRgI0B(FitFunction):
    name = "Gaussian distribution of core-shell spheres, parametrized with Rg and I0, branch B"

    description = "Spherical core-shell nanoparticles, with Rg and I0, branch B"

    arguments = [('I0', 'Intensity extrapolated to zero'),
                 ('background', 'Constant background'),
                 ('Rg', 'Radius of gyration'),
                 ('rcore', 'Mean core radius'),
                 ('sigmacore', 'HWHM of the core radius'),
                 ('tshell', 'Shell thickness'),
                 ]

    def function(self, x, I0, background, Rg, rcore, sigmacore, tshell):
        R5=rcore**5
        R3=rcore**3
        Rpt3= (rcore+tshell)**3
        Rpt5 = (rcore+tshell)**5
        Rg253=5/3*Rg**2
        rhocoredivrhoshell=(Rpt5-R5-Rg253*(Rpt3-R3))/(Rg253*R3-R5)
        rhoshell = -3*I0**0.5/4/np.pi/(Rpt3-R3+R3*rhocoredivrhoshell)
        rhocore = rhoshell*rhocoredivrhoshell
        return F2GaussianCoreShellSphereDistribution(x, rcore, tshell, sigmacore, rhocore, rhoshell) + background

    def draw_representation(self, fig, x, I0, background, Rg, rcore, sigmacore, tshell):
        R5=rcore**5
        R3=rcore**3
        Rpt3= (rcore+tshell)**3
        Rpt5 = (rcore+tshell)**5
        Rg253=5/3*Rg**2
        fig.clear()
        ax=fig.add_subplot(1,1,1)
        assert isinstance(ax, Axes)
        p=Ellipse((0,0),2*(rcore+tshell),2*(rcore+tshell),color='yellow')
        ax.add_patch(p)
        p=Ellipse((0,0),2*(rcore),2*(rcore), color='green')
        ax.add_patch(p)
        rhocoredivrhoshell=(Rpt5-R5-Rg253*(Rpt3-R3))/(Rg253*R3-R5)
        rhoshell = -3*I0**0.5/4/np.pi/(Rpt3-R3+R3*rhocoredivrhoshell)
        rhocore = rhoshell*rhocoredivrhoshell
        ax.autoscale_view(True, True, True)
        ax.text(0.05,0.95,
                '$R_\mathrm{{core}}$: {}\n'
                '$T_\mathrm{{shell}}$: {}\n'
                '$\\rho_\mathrm{{core}}$: {}\n'
                '$\\rho_\mathrm{{shell}}$: {}\n'
                '$I_0$: {}\n'
                '$R_g$: {}\n'.format(rcore, tshell, rhocore, rhoshell, I0, Rg),
                transform=ax.transAxes,ha='left',va='top')
        fig.canvas.draw()
