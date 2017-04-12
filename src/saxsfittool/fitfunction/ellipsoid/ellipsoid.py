import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from .c_ellipsoid2 import AsymmetricEllipsoidalShell
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
