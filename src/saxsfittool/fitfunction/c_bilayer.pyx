import numpy as np
cimport numpy as np
from libc.math cimport exp, sin, cos

np.import_array()

cdef double FGaussianLayer(double q, double rho, double r, double sigma):
    """Scattering form factor amplitude of a radial Gaussian electron density
    distribution."""
    return 2 * rho * sigma * exp(-q * q * sigma * sigma / 2) * (r * sin(q * r) + sigma * sigma * q * cos(q * r)) / q

cdef double FFiveGaussSymmetricHeadBilayer(
        double q, double rtail, double rhoguestin, double rhohead,
        double rhoguestout, double deltarguestin, double deltarhead,
        double deltarguestout, double sigmaguestin, double sigmahead,
        double sigmatail, double sigmaguestout):
    return FGaussianLayer(q, rhoguestin, rtail - deltarguestin, sigmaguestin) + \
           FGaussianLayer(q, rhohead, rtail - deltarhead, sigmahead) + \
           FGaussianLayer(q, -1.0, rtail, sigmatail) + \
           FGaussianLayer(q, rhohead, rtail + deltarhead, sigmahead) + \
           FGaussianLayer(q, rhoguestout, rtail + deltarguestout, sigmaguestout)

def F2FiveGaussSymmetricHeadBilayer(np.ndarray[np.double_t, ndim=1] q not None, double R, double dR, double d,
                                    double rhoguestin, double rhohead, double rhoguestout, double deltarguestin,
                                    double deltarhead, double deltarguestout, double sigmaguestin, double sigmahead,
                                    double sigmatail, double sigmaguestout, Py_ssize_t Nbilayers=1,
                                    Py_ssize_t NRdistrib=100):
    out = np.empty(len(q), np.double)
    cdef np.broadcast it = np.broadcast(q, out)
    cdef double q_, intensity_averaged, r, weight, sumweight, intensity_singleobject
    cdef Py_ssize_t i, j
    while np.PyArray_MultiIter_NOTDONE(it):
        q_ = (<double*> np.PyArray_MultiIter_DATA(it, 0))[0]
        intensity_averaged = 0
        sumweight=0
        for j in range(NRdistrib):
            r = R - dR * 3 + (j / (NRdistrib - 1.0)) * 6*dR
            w = exp(-(r - R) * (r - R) / (2 * dR * dR))
            sumweight+=w
            intensity_singleobject=0
            for i in range(Nbilayers):
                intensity_singleobject += FFiveGaussSymmetricHeadBilayer(
                    q_, r + d * i, rhoguestin, rhohead, rhoguestout, deltarguestin,
                    deltarhead, deltarguestout, sigmaguestin, sigmahead, sigmatail,
                    sigmaguestout)
            intensity_averaged += w*intensity_singleobject**2
        (<double*> np.PyArray_MultiIter_DATA(it, 1))[0] = intensity_averaged/sumweight
        np.PyArray_MultiIter_NEXT(it)
    return out
