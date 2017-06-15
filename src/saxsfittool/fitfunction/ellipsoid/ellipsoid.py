import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from .c_ellipsoid2 import AsymmetricEllipsoidalShell, EllipsoidalShellWithSizeDistribution
from .c_gauss_ellipsoid import I0Rgfromrho, rhofromI0Rg, F2GaussianEllipsoid
from ..core import FitFunction


class F2AsymmetricCoreShellEllipsoidWithBackgroundI0Rg(FitFunction):
    name = 'Rotational core-shell ellipsoid'

    arguments = [('I0', 'Intensity extrapolated to zero'),
                 ('Rg', 'Radius of gyration'),
                 ('a', 'Principal semi-axis of the core'),
                 ('b', 'Equatorial semi-axis of the core'),
                 ('ta', 'Shell thickness (along the principal axis)'),
                 ('tb', 'Shell thickness (along the equatorial axes)'),
                 ('bg', 'Constant background')]

    unfittable_parameters = [
        ('ninteg','Number of points for numerical averaging', 2,100000,100),
    ]

    description = "Scattering intensity of an asymmetric rotational core-shell ellipsoid with additional constant background"

    def _get_rhos(self, I0,Rg,a,b,ta,tb):
        btb=b+tb
        ata=a+ta
        btb2ata = btb**2*ata
        b2a = b**2*a
        rhocoredivrhoshell = (ata**3*btb**2+2*ata*btb**4-5*Rg**2*ata*btb**2-a**3*b**2-2*a*b**4+5*a*b**2*Rg**2)/(5*a*b**2*Rg**2-a**3*b**2-2*a*b**4)
        eta_shell = 3*I0**0.5/4/np.pi/(rhocoredivrhoshell*b2a+btb2ata-b2a)
        eta_core = rhocoredivrhoshell*eta_shell
        return eta_core, eta_shell

    def function(self, x, I0, Rg, a, b, ta, tb, C, ninteg):
        eta_core, eta_shell = self._get_rhos(I0, Rg, a, b, ta, tb)
        return AsymmetricEllipsoidalShell(x, eta_core, eta_shell, a, b, ta, tb) + C

    def draw_representation(self, fig, x, I0, Rg, a, b, ta, tb, C, ninteg):
        eta_core, eta_shell = self._get_rhos(I0, Rg, a, b, ta, tb)
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
        ax.axis('equal')
        fig.canvas.draw()

class F2SymmetricCoreShellEllipsoidWithBackgroundI0RgA(F2AsymmetricCoreShellEllipsoidWithBackgroundI0Rg):
    name = 'Rotational core-shell ellipsoid, symmetric shell'

    arguments = [('I0', 'Intensity extrapolated to zero'),
                 ('Rg', 'Radius of gyration'),
                 ('a', 'Principal semi-axis of the core'),
                 ('b', 'Equatorial semi-axis of the core'),
                 ('t', 'Shell thickness (along both axes)'),
                 ('bg', 'Constant background')]

    description = "Scattering intensity of a symmetric rotational core-shell ellipsoid with additional constant background"

    def function(self, x, I0, Rg, a, b, t, C, ninteg):
        return super().function(x,I0,Rg,a,b,t,t,C,ninteg)

    def draw_representation(self, fig, x, I0, Rg, a, b, t, C, ninteg):
        return super().draw_representation(fig, x, I0, Rg, a, b, t, t, C, ninteg)

class F2EllipsoidalShellWithSizeDistribution(F2AsymmetricCoreShellEllipsoidWithBackgroundI0Rg):
    name = 'Ellipsoidal core-shell particle with size distribution'

    arguments = [('I0', 'Intensity extrapolated to zero'),
                 ('Rg', 'Radius of gyration'),
                 ('a', 'Rotational semi-axis of the core'),
                 ('b/a', 'Ratio of the equatorial and the rotational semi-axes of the core'),
                 ('sigma(b/a)','HWHM of the Gaussian distribution placed on b/a'),
                 ('ta', 'Shell thickness along the rotational axis'),
                 ('tb/ta', 'Ratio of the shell thicknesses along the rotational and the equatorial semi-axes'),
                 ('sigma(tb/ta)','HWHM of the Gaussian distribution placed on tb/ta'),
                 ('bg', 'Constant background')]

    unfittable_parameters = [
        ('Nintorientation','Number of points for numerical averaging of the macroscopic orientation', 2,100000,100),
        ('Nintanisometrycore', 'Number of points for numerical averaging of the core anisometry',2,100000,100),
        ('Nintanisometryshell', 'Number of points for numerical averaging of the shell anisometry',2,100000,100),
    ]

    def function(self, x, I0, Rg, a, bdiva_mean, bdiva_sigma, ta, tbdivta_mean, tbdivta_sigma, C, Nintorientation, Nintanisometrycore, Nintanisometryshell):
        rhocore, rhoshell = self._get_rhos(I0, Rg, a,bdiva_mean*a, ta, tbdivta_mean*ta)
        return EllipsoidalShellWithSizeDistribution(x, rhocore, rhoshell, a, bdiva_mean, bdiva_sigma, ta, tbdivta_mean, tbdivta_sigma, int(Nintorientation), int(Nintanisometrycore), int(Nintanisometryshell))+C

    def draw_representation(self, fig, x, I0, Rg, a, bdiva_mean, bdiva_sigma, ta, tbdivta_mean, tbdivta_sigma, C, Nintorientation, Nintanisometrycore, Nintanisometryshell):
        super().draw_representation(fig, x, I0, Rg, a, bdiva_mean*a, ta, tbdivta_mean*ta, C, Nintorientation)


class GaussianEllipsoid(FitFunction):
    name = 'Gaussian shell in an ellipsoid'

    arguments = [('I0', 'Intensity at q=0'),
                 ('Rg', 'Radius of gyration'),
                 ('a', 'Principal semi-axis of the core'),
                 ('b', 'Equatorial semi-axis of the core'),
                 ('sigmain', 'Core-side HWHM of the Gauss shell layer'),
                 ('sigmaout', 'Solvent-side HWHM of the Gauss shell layer'),
                 ('bg', 'Constant background')]

    description = "Scattering intensity of a Gaussian shell on an ellipsoid of rotation"

    unfittable_parameters = [
        ('Nintr','Number of points for numerical integration with respect to "r"', 10,100000,100),
        ('Nintu','Number of points for numerical integration with respect to "u"', 10,100000,100),
        ('NintU','Number of points for numerical integration with respect to "U"', 10,100000,100),
    ]

    def function(self, x, I0, Rg, a, b, sigmain, sigmaout, bg, Nintr, Nintu, NintU):
        rhocore, rhoshell = rhofromI0Rg(I0, Rg, a, b, sigmain, sigmaout)
        return F2GaussianEllipsoid(x, a, b, rhocore, rhoshell, sigmain, sigmaout, int(Nintr), int(Nintu), int(NintU))+bg

class GaussianEllipsoidSymmetricHead(FitFunction):
    name = 'Gaussian shell in an ellipsoid, symmetric head'

    arguments = [('I0', 'Intensity at q=0'),
                 ('Rg', 'Radius of gyration'),
                 ('a', 'Principal semi-axis of the core'),
                 ('b', 'Equatorial semi-axis of the core'),
                 ('sigma', 'HWHM of the Gauss shell layer'),
                 ('bg', 'Constant background')]

    description = "Scattering intensity of a Gaussian shell on an ellipsoid of rotation. The head is symmetric"

    unfittable_parameters = [
        ('Nintr','Number of points for numerical integration with respect to "r"', 10,100000,100),
        ('Nintu','Number of points for numerical integration with respect to "u"', 10,100000,100),
        ('NintU','Number of points for numerical integration with respect to "U"', 10,100000,100),
    ]

    def function(self, x, I0, Rg, a, b, sigma, bg, Nintr, Nintu, NintU):
        rhocore, rhoshell = rhofromI0Rg(I0, Rg, a, b, sigma, sigma)
        return F2GaussianEllipsoid(x, a, b, rhocore, rhoshell, sigma, sigma, int(Nintr), int(Nintu), int(NintU))+bg
