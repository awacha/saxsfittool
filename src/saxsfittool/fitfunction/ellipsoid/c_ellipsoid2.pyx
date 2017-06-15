#cython: cdivision=True
#cython: embedsignature=True
#cython: nonecheck=False
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc cimport math
from libc.float cimport DBL_EPSILON
from cpython.mem cimport PyMem_Malloc, PyMem_Free

np.import_array()

cdef inline double j1divx(double x) nogil:
    return (math.sin(x) - x * math.cos(x)) / x ** 3

cdef inline double V(double a, double b) nogil:
    return 4 * math.M_PI * a * b ** 2 / 3

cdef inline double gaussian(double x, double x0, double sigma) nogil:
    return math.exp(-(x-x0)*(x-x0)/(2*sigma*sigma))

cdef inline double AmplitudeShellOrientedScalar(
        double q, double rhocore, double rhoshell,
        double a, double b, double ta, double tb, double mu) nogil:
    cdef:
        double ata
        double btb
        double mu2
    ata = a + ta
    btb = b + tb
    mu2 = mu ** 2
    return ((rhocore - rhoshell) * V(a, b) * 3 * j1divx(q * (a ** 2 * mu2 + b ** 2 * (1 - mu2)) ** 0.5) +
            rhoshell * V(ata, btb) * 3 * j1divx(q * (ata ** 2 * mu2 + btb ** 2 * (1 - mu2)) ** 0.5))

cdef inline double IntensityShellScalar(
        double q, double rhocore, double rhoshell,
        double a, double b, double ta, double tb, Py_ssize_t N=1000) nogil:
    cdef:
        double I
        Py_ssize_t i
    I = 0
    for i in range(0, N):
        I += AmplitudeShellOrientedScalar(q, rhocore, rhoshell, a, b, ta, tb, i / (N - 1.0))**2
    return I / N

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

cdef double IntensityShellScalarSmearedBdivA(
        double q, double rhocore, double rhoshell,
        double a, double bdiva_mean, double bdiva_sigma,
        double ta, double tbdivta_mean, double tbdivta_sigma,
        Py_ssize_t Nintorientation=100, Py_ssize_t Nintanisometrycore=100, Py_ssize_t Nintanisometryshell = 100) nogil:
    cdef:
        Py_ssize_t ianisocore, ianisoshell
        double I = 0
        double I1 = 0
        double w=0
        double w1=0
        double W=0
        double W1=0
        double alpha=0
        double beta=0
        double dalpha = 0 # alpha is b/a
        double dbeta = 0 # beta is tb/ta
    if (math.fabs(bdiva_sigma) < DBL_EPSILON) and (math.fabs(tbdivta_sigma)<DBL_EPSILON):
        return IntensityShellScalar(q,rhocore,rhoshell,a,bdiva_mean*a, ta, tbdivta_mean*ta, Nintorientation)
    elif math.fabs(bdiva_sigma) < DBL_EPSILON:
        dbeta = (6*tbdivta_sigma)/(Nintanisometryshell-1.)
        I=0
        for ianisoshell in range(0, Nintanisometryshell):
            beta = tbdivta_mean-3*tbdivta_sigma+dbeta*ianisoshell
            W=math.exp(-(6.*ianisoshell/(Nintanisometryshell-1.)-3)**2/2.)
            w+=W
            I+=W*IntensityShellScalar(q, rhocore, rhoshell, a, a*bdiva_mean, ta, ta*beta, Nintorientation)
        return I/w
    elif math.fabs(tbdivta_sigma) < DBL_EPSILON:
        dalpha = (6*bdiva_sigma)/(Nintanisometrycore-1.)
        I=0
        for ianisocore in range(0, Nintanisometrycore):
            alpha = bdiva_mean-3*bdiva_sigma+dalpha*ianisocore
            W=math.exp(-(6.*ianisocore/(Nintanisometrycore-1.)-3)**2/2.)
            w+=W
            I+=W*IntensityShellScalar(q, rhocore, rhoshell, a, a*alpha, ta, ta*tbdivta_mean, Nintorientation)
        return I/w
    else:
        dalpha = (6*bdiva_sigma)/(Nintanisometrycore-1.)
        dbeta = (6*tbdivta_sigma)/(Nintanisometryshell-1.)
        I=0
        w=0
        for ianisocore in range(0, Nintanisometrycore):
            alpha = bdiva_mean-3*bdiva_sigma+dalpha*ianisocore
            I1=0
            w1=0
            W=math.exp(-(6.*ianisocore/(Nintanisometrycore-1.)-3)**2/2.)
            for ianisoshell in range(0,Nintanisometryshell):
                beta = tbdivta_mean-3*tbdivta_sigma+dbeta*ianisoshell
                W1=math.exp(-(6.*ianisoshell/(Nintanisometryshell-1.)-3)**2/2.)
                w1+=W1
                I1+=W1*IntensityShellScalar(q, rhocore, rhoshell, a, a*alpha, ta, ta*beta, Nintorientation)
            I += I1/w1*W
            w+=W
        return I/w

def EllipsoidalShellWithSizeDistribution(
        np.ndarray[np.double_t, ndim=1] q not None,
        double rhocore, double rhoshell,
        double a, double bdiva_mean, double bdiva_sigma,
        double ta, double tbdivta_mean, double tbdivta_sigma,
        Py_ssize_t Nintorientation=100, Py_ssize_t Nintanisometrycore=100,
        Py_ssize_t Nintanisometryshell=100):
    cdef:
        np.ndarray[np.double_t, ndim=1] out
        double *my_out
        double *my_q
        Py_ssize_t i, N
    N=len(q)
    my_out = <double*>PyMem_Malloc(sizeof(double)*N)
    my_q = <double*>PyMem_Malloc(sizeof(double)*N)
    for i in range(N):
        my_q[i]=q[i]
    for i in prange(0,N, nogil=True):
        my_out[i] = IntensityShellScalarSmearedBdivA(
            my_q[i], rhocore, rhoshell,
            a, bdiva_mean, bdiva_sigma,
            ta, tbdivta_mean, tbdivta_sigma,
            Nintorientation, Nintanisometrycore, Nintanisometryshell
        )
    out = np.empty_like(q, np.double)
    for i in range(N):
        out[i]=my_out[i]
    PyMem_Free(my_q)
    PyMem_Free(my_out)
    return out
