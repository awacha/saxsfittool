import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from .c_coreshell import F2GaussianCoreShellSphereDistribution
from ..core import FitFunction


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

class CoreShellSphereGaussianDistributionRgI0(FitFunction):
    name = "Gaussian distribution of core-shell spheres with Rg and I0"

    description = "Spherical core-shell nanoparticles, with Rg and I0"

    arguments = [('I0', 'Intensity extrapolated to zero'),
                 ('background', 'Constant background'),
                 ('Rg', 'Radius of gyration'),
                 ('rcore', 'Mean core radius'),
                 ('sigmacore', 'HWHM of the core radius'),
                 ('tshell', 'Shell thickness'),
                 ]

    unfittable_parameters = [('Ninteg', 'Number of points for numerical integration', 2,100000,100),
                             ]

    def _get_rhos(self, I0, Rg, rcore, tshell):
        R5=rcore**5
        R3=rcore**3
        Rpt3= (rcore+tshell)**3
        Rpt5 = (rcore+tshell)**5
        Rg253=5/3*Rg**2
        rhocoredivrhoshell=(Rpt5-R5-Rg253*(Rpt3-R3))/(Rg253*R3-R5)
        rhoshell = 3*I0**0.5/4/np.pi/(Rpt3-R3+R3*rhocoredivrhoshell)
        rhocore = rhoshell*rhocoredivrhoshell
        return rhocore, rhoshell

    def function(self, x, I0, background, Rg, rcore, sigmacore, tshell, ninteg):
        rhocore, rhoshell = self._get_rhos(I0, Rg, rcore, tshell)
        return F2GaussianCoreShellSphereDistribution(x, rcore, tshell, sigmacore, rhocore, rhoshell) + background

    def draw_representation(self, fig, x, I0, background, Rg, rcore, sigmacore, tshell, ninteg):
        rhocore, rhoshell = self._get_rhos(I0, Rg, rcore, tshell)
        fig.clear()
        ax=fig.add_subplot(1,1,1)
        assert isinstance(ax, Axes)
        p=Ellipse((0,0),2*(rcore+tshell),2*(rcore+tshell),color='yellow')
        ax.add_patch(p)
        p=Ellipse((0,0),2*(rcore),2*(rcore), color='green')
        ax.add_patch(p)
        ax.autoscale_view(True, True, True)
        ax.text(0.05,0.95,
                '$R_\mathrm{{core}}$: {}\n'
                '$T_\mathrm{{shell}}$: {}\n'
                '$\\rho_\mathrm{{core}}$: {}\n'
                '$\\rho_\mathrm{{shell}}$: {}\n'
                '$I_0$: {}\n'
                '$R_g$: {}\n'.format(rcore, tshell, rhocore, rhoshell, I0, Rg),
                transform=ax.transAxes,ha='left',va='top')
        ax.axis('equal')
        fig.canvas.draw()
