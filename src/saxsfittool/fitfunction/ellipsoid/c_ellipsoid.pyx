import cython
cimport cython
import numpy as np
cimport numpy as np
from libc cimport math

np.import_array()

cdef double j1divx(double x):
    return (math.sin(x)-x*math.cos(x))/x**3

@cython.cdivision
def FEllipsoidalShell_oriented(np.ndarray[np.float_t, ndim=1] q not None, double etacore, double etashell,
                      double etasolvent, double a, double b, double t, double mu):
    """Calculate the form factor amplitude of a core-shell structure made of
    two concentric rotational ellipsoids. Based on $3.2.3 in the SASFit manual
    (version 2nd June, 2014, https://kur.web.psi.ch/sans1/SANSSoft/sasfit.pdf,
    retrieved 5th October, 2016.)

    Inputs:
        q: float or np.ndarray
            the scattering variable
        etacore: float
            the scattering length density of the core
        etashell: float
            the scattering length density of the shell
        etasolvent: float
            the scattering length density of the solvent
        a: float
            principal semi-axis of the ellipsoidal core
        b: float
            equatorial semi-axis of the ellipsoidal core
        t: float
            thickness of the shell
        mu: float
            orientation parameter ranging from 0 to 1
    """
    cdef:
        double Vc, Vt
        double xc, xt
        double q_, f_
        np.ndarray[np.double_t, ndim=1] out
        Py_ssize_t i, N
    out=np.empty(len(q))
    xcfactor=(a**2*mu**2+b**2*(1-mu**2))**0.5
    xtfactor=((a+t)**2*mu**2+(b+t)**2*(1-mu**2))**0.5
    Vc=4/3*math.M_PI*a*b**2
    Vt=4/3*math.M_PI*(a+t)*(b+t)**2
    N=len(q)
    for i in range(0, N):
        q_=q[i]
        f_=(etacore-etashell)*Vc*(3*j1divx(q_*xcfactor))+(etashell-etasolvent)*Vt*(3*j1divx(xtfactor*q_))
        out[i]=f_
    return out

@cython.cdivision
cdef double FEllipsoidalShell_oriented_scalar(double q, double etacore, double etashell,
                      double etasolvent, double a, double b, double t, double mu):
    """Calculate the form factor amplitude of a core-shell structure made of
    two concentric rotational ellipsoids. Based on $3.2.3 in the SASFit manual
    (version 2nd June, 2014, https://kur.web.psi.ch/sans1/SANSSoft/sasfit.pdf,
    retrieved 5th October, 2016.)

    Inputs:
        q: float
            the scattering variable
        etacore: float
            the scattering length density of the core
        etashell: float
            the scattering length density of the shell
        etasolvent: float
            the scattering length density of the solvent
        a: float
            principal semi-axis of the ellipsoidal core
        b: float
            equatorial semi-axis of the ellipsoidal core
        t: float
            thickness of the shell
        mu: float
            orientation parameter ranging from 0 to 1
    """
    cdef:
        double Vc, Vt
        double xc, xt
        double out=0
    xcfactor=(a**2*mu**2+b**2*(1-mu**2))**0.5
    xtfactor=((a+t)**2*mu**2+(b+t)**2*(1-mu**2))**0.5
    Vc=4/3*math.M_PI*a*b**2
    Vt=4/3*math.M_PI*(a+t)*(b+t)**2
    return (etacore-etashell)*Vc*(3*j1divx(q*xcfactor))+(etashell-etasolvent)*Vt*(3*j1divx(xtfactor*q))

@cython.cdivision
def F2EllipsoidalShell(np.ndarray[np.float_t, ndim=1] q not None, double etacore, double etashell,
                      double etasolvent, double a, double b, double t, Py_ssize_t Nmu=1000):
    cdef:
        Py_ssize_t i
        double mu
        np.ndarray[np.double_t, ndim=1] out
    out=FEllipsoidalShell_oriented(q, etacore, etashell, etasolvent, a, b, t, 0.0)
    for i in range(0, Nmu):
        mu=1/(Nmu)*(i+1)
        out += FEllipsoidalShell_oriented(q, etacore, etashell, etasolvent, a, b, t, mu)**2
    return out/Nmu

@cython.cdivision
def F2EllipsoidalShell_scalar(double q, double etacore, double etashell,
                      double etasolvent, double a, double b, double t, Py_ssize_t Nmu=1000):
    cdef:
        Py_ssize_t i
        double mu
        double out=0
    for i in range(0, Nmu):
        mu=1/(Nmu)*(i+1)
        out += FEllipsoidalShell_oriented_scalar(q, etacore, etashell, etasolvent, a, b, t, mu)**2
    return out/Nmu

@cython.cdivision
def FAsymmetricEllipsoidalShell_oriented(np.ndarray[np.float_t, ndim=1] q not None, double etacore, double etashell,
                      double etasolvent, double a, double b, double ta, double tb, double mu):
    """Calculate the form factor amplitude of a core-shell structure made of
    two concentric rotational ellipsoids. Based on $3.2.3 in the SASFit manual
    (version 2nd June, 2014, https://kur.web.psi.ch/sans1/SANSSoft/sasfit.pdf,
    retrieved 5th October, 2016.)

    Inputs:
        q: float or np.ndarray
            the scattering variable
        etacore: float
            the scattering length density of the core
        etashell: float
            the scattering length density of the shell
        etasolvent: float
            the scattering length density of the solvent
        a: float
            principal semi-axis of the ellipsoidal core
        b: float
            equatorial semi-axis of the ellipsoidal core
        ta: float
            thickness of the shell in the "a" direction
        tb: float
            thickness of the shell in the "b" direction
        mu: float
            orientation parameter ranging from 0 to 1
    """
    cdef:
        double Vc, Vt
        double xc, xt
        double q_, f_
        np.ndarray[np.double_t, ndim=1] out
        Py_ssize_t i, N
    out=np.empty(len(q))
    xcfactor=(a**2*mu**2+b**2*(1-mu**2))**0.5
    xtfactor=((a+tb)**2*mu**2+(b+tb)**2*(1-mu**2))**0.5
    Vc=4/3*math.M_PI*a*b**2
    Vt=4/3*math.M_PI*(a+ta)*(b+tb)**2
    N=len(q)
    for i in range(0, N):
        q_=q[i]
        f_=(etacore-etashell)*Vc*(3*j1divx(q_*xcfactor))+(etashell-etasolvent)*Vt*(3*j1divx(xtfactor*q_))
        out[i]=f_
    return out

@cython.cdivision
cdef double FAsymmetricEllipsoidalShell_oriented_scalar(double q, double etacore, double etashell,
                      double etasolvent, double a, double b, double ta, double tb, double mu):
    """Calculate the form factor amplitude of a core-shell structure made of
    two concentric rotational ellipsoids. Based on $3.2.3 in the SASFit manual
    (version 2nd June, 2014, https://kur.web.psi.ch/sans1/SANSSoft/sasfit.pdf,
    retrieved 5th October, 2016.)

    Inputs:
        q: float
            the scattering variable
        etacore: float
            the scattering length density of the core
        etashell: float
            the scattering length density of the shell
        etasolvent: float
            the scattering length density of the solvent
        a: float
            principal semi-axis of the ellipsoidal core
        b: float
            equatorial semi-axis of the ellipsoidal core
        ta: float
            thickness of the shell in the a direction
        tb: float
            thickness of the shell in the b direction
        mu: float
            orientation parameter ranging from 0 to 1
    """
    cdef:
        double Vc, Vt
        double xc, xt
        double out=0
    xcfactor=(a**2*mu**2+b**2*(1-mu**2))**0.5
    xtfactor=((a+ta)**2*mu**2+(b+tb)**2*(1-mu**2))**0.5
    Vc=4/3*math.M_PI*a*b**2
    Vt=4/3*math.M_PI*(a+ta)*(b+tb)**2
    return (etacore-etashell)*Vc*(3*j1divx(q*xcfactor))+(etashell-etasolvent)*Vt*(3*j1divx(xtfactor*q))

@cython.cdivision
def F2AsymmetricEllipsoidalShell(np.ndarray[np.float_t, ndim=1] q not None, double etacore, double etashell,
                      double etasolvent, double a, double b, double ta, double tb, Py_ssize_t Nmu=1000):
    cdef:
        Py_ssize_t i
        double mu
        np.ndarray[np.double_t, ndim=1] out
    out=FAsymmetricEllipsoidalShell_oriented(q, etacore, etashell, etasolvent, a, b, ta, tb, 0.0)**2
    for i in range(0, Nmu):
        mu=1/(Nmu)*(i+1)
        out += FAsymmetricEllipsoidalShell_oriented(q, etacore, etashell, etasolvent, a, b, ta, tb, mu)**2
    return out/Nmu

@cython.cdivision
def F2AsymmetricEllipsoidalShell_scalar(double q, double etacore, double etashell,
                      double etasolvent, double a, double b, double ta, double tb, Py_ssize_t Nmu=1000):
    cdef:
        Py_ssize_t i
        double mu
        double out=0
    for i in range(0, Nmu):
        mu=1/(Nmu)*(i+1)
        out += FAsymmetricEllipsoidalShell_oriented_scalar(q, etacore, etashell, etasolvent, a, b, ta, tb, mu)**2
    return out/Nmu
