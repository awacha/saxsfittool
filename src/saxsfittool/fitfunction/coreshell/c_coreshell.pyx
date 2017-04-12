import numpy as np
cimport numpy as np
from libc.math cimport exp, sin, cos, M_PI

np.import_array()

cdef double gaussian(double x, double x0, double sigma):
    return exp(-(x - x0) ** 2 / sigma ** 2)

cdef double phisphere(double q, double R):
    cdef double qR = q * R
    return 3 / qR ** 3 * (sin(qR) - qR * cos(qR))

cdef double Vsphere(double R):
    return 4 * M_PI * R ** 3 / 3.

cdef double f2gaussiancoreshellspheredistribution(double q,
                                           double rin, double shellthickness,
                                           double sigmarin, double rhoin,
                                           double rhoshell, unsigned long N=1000):
    cdef:
        double r = rin - 3 * sigmarin
        double dr = 6 * sigmarin / (N - 1)
        double intensity = 0, weight = 0, w = 0
    while r <= rin + 3 * sigmarin:
        w = gaussian(r, rin, sigmarin)
        intensity += ((rhoin-rhoshell)*Vsphere(r)*phisphere(q, r)+rhoshell*Vsphere(r+shellthickness)*phisphere(q,r+shellthickness)) ** 2 * w
        weight += w
        r += dr
    return intensity / weight


def F2GaussianCoreShellSphereDistribution(
        np.ndarray[np.double_t] q, double rin, double shellthickness,
        double sigmarin, double rhoin, double rhoshell, unsigned long N=1000):
    out = np.empty_like(q, np.double)
    cdef:
        np.broadcast it = np.broadcast(q, out)
        double q_

    while np.PyArray_MultiIter_NOTDONE(it):
        q_ = (<double*> np.PyArray_MultiIter_DATA(it, 0))[0]
        (<double*> np.PyArray_MultiIter_DATA(it, 1))[0] = f2gaussiancoreshellspheredistribution(
            q_, rin, shellthickness, sigmarin, rhoin, rhoshell, N)
        np.PyArray_MultiIter_NEXT(it)
    return out
