cimport
numpy as np
import cython
import numpy as np
from libc cimport

np.import_array()

cdef double j1divx(double x):
    return (math.sin(x) - x * math.cos(x)) / x ** 3

cdef double V(double a, double b):
    return 4 * math.M_PI * a * b ** 2 / 3

@cython.cdivision
cdef double AmplitudeShellOrientedScalar(
        double q, double rhocore, double rhoshell,
        double a, double b, double ta, double tb, double mu):
    cdef:
        double ata
        double btb
        double mu2
    ata = a + ta
    btb = b + tb
    mu2 = mu ** 2
    return ((rhocore - rhoshell) * V(a, b) * 3 * j1divx(q * (a ** 2 * mu2 + b ** 2 * (1 - mu2)) ** 0.5) +
            rhoshell * V(ata, btb) * 3 * j1divx(q * (ata ** 2 * mu2 + btb ** 2 * (1 - mu2)) ** 0.5))

@cython.cdivision
cdef double IntensityShellScalar(
        double q, double rhocore, double rhoshell,
        double a, double b, double ta, double tb, Py_ssize_t N=1000):
    cdef:
        double I
        Py_ssize_t i
    I = 0
    for i in range(0, N):
        I += AmplitudeShellOrientedScalar(q, rhocore, rhoshell, a, b, ta, tb, i / (N - 1.0))**2
    return I / N

@cython.cdivision
def AsymmetricEllipsoidalShell(
        np.ndarray[np.double_t, ndim=1] q not None,
        double rhocore, double rhoshell, double a,
        double b, double ta, double tb, Py_ssize_t N=1000):
    out = np.empty_like(q, np.double)
    cdef:
        np.broadcast it = np.broadcast(q, out)
        double q_
    while np.PyArray_MultiIter_NOTDONE(it):
        q_ = (<double*> np.PyArray_MultiIter_DATA(it, 0))[0]
        (<double*> np.PyArray_MultiIter_DATA(it, 1))[0] = IntensityShellScalar(q_, rhocore, rhoshell, a,b,ta,tb,N)
        np.PyArray_MultiIter_NEXT(it)
    return out
