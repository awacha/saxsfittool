import numpy as np
from matplotlib.figure import Figure

from .c_bilayer import F2FiveGaussSymmetricHeadBilayer
from ..core import FitFunction


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

