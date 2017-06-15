# cython: cdivision=True
# cython: embedsignature=True
# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport exp, cos, asinh, asin, atanh, atan, M_PI, sqrt, log, isnan
from cython.parallel cimport prange
from cpython.mem cimport PyMem_Malloc, PyMem_Free

np.import_array()

cdef extern from "<math.h>" nogil:
    double j0(double x)

cdef inline double gaussian(double x, double x0, double sigma) nogil:
    return exp(-(x-x0)**2/(2*sigma**2)) if sigma!=0 else <double>(x==x0)

cdef inline double rho(double r, double u, double a, double b,
                double rhocore, double rhoshell, double sigmain, double sigmaout) nogil:
    """Calculate the electron density of an ellipsoidal gaussian shell.
    
    Inputs:
        r: distance from the origin
        u: cos(theta) where theta is the polar angle in the spherical coordinate system
        a: half axis of the ellipsoid of revolution, parallel to the axis of revolution (z)
        b: half axis of the ellipsoid of revolution, orthogonal to the axis of revolution (z)
        rhocore: electron density at r=0
        rhoshell: highest electron density of the shell
        sigmain: HWHM of the inner half Gaussian
        sigmaout: HWHM of the outer half Gaussian
        
    Outputs:
        the electron density value
    """
    cdef:
        double R = 1/((1-u**2)/b**2 +u**2/a**2 )**0.5
    return ((rhocore + (rhoshell-rhocore)*gaussian(r,R,sigmain)) if (r<R) else (rhoshell*gaussian(r,R,sigmaout)))
#    if r<R:
#        return rhocore + (rhoshell-rhocore)*gaussian(r,R,sigmain)
#    else:
#        return rhoshell*gaussian(r,R,sigmaout)

# the functions below use integration by the trapezoid method using the following approach:
#
# given a function f(), which has to be integrated over [a,b] with Nint steps:
#
# result = 0
# for i in range(Nint):
#     x = a+(b-a)/(Nint-1)
#     result += f(x)
# result = (b-a)/(Nint-1) * (result - 0.5*(f(a)+f(b)))
#

cdef double F2(double q, double U, double a, double b,
               double rhocore, double rhoshell, double sigmain, double sigmaout,
               Py_ssize_t Nintr=100, Py_ssize_t Nintu=100) nogil:
    """Calculate the scattering intensity of an electron density of an ellipsoidal gaussian shell.
    Everything is in a spherical polar coordinate system with the z axis parallel with
    the axis of revolution of the ellipsoid.
    
    Inputs:
        q: magnitude of the scattering variable
        U: cosine of the polar angle of q.
        a: half axis of the ellipsoid of revolution, parallel to the axis of revolution (z)
        b: half axis of the ellipsoid of revolution, orthogonal to the axis of revolution (z)
        rhocore: electron density at r=0
        rhoshell: highest electron density of the shell
        sigmain: HWHM of the inner half Gaussian
        sigmaout: HWHM of the outer half Gaussian
        
    Outputs:
        the scattered intensity F^2(q, U)
    """
    cdef:
        double retval=0
        double retval_r=0
        double u=0
        double du=1./(Nintu-1)
        double v=0
        double r=0
        double qxy=q*(1-U**2)
        double qz=q*U
        double Rmax=0
        double dr=0
        Py_ssize_t ir=0
        Py_ssize_t iu=0
        double int_r_u0 = 0
        double int_r_u1 = 0
        double f
    for iu in range(0,Nintu):
        u=iu*du
        v=(1-u**2)**0.5
        Rmax = 1/((1-u**2)/b**2+u**2/a**2)**0.5 +sigmaout*3
        dr=Rmax/(Nintr-1.)
        if isnan(Rmax):
            with gil:
                print('Rmax is NaN: qxy={}, qz={}, u={}'.format(qxy,qz,u))
        if isnan(v):
            with gil:
                print('v is NaN: qxy={}, qz={}, u={}'.format(qxy,qz,u))
        retval_r=0
        for ir in range(1, Nintr): # skip r=0, the integrand will be zero there.
            r=ir*dr
            f=r**2*rho(r, u, a, b, rhocore, rhoshell, sigmain, sigmaout)*j0(r*qxy*v)*cos(r*qz*u)
            retval_r+=f
        # we do not need to subtract 0.5*(f(a) + f(b)) since f(a)==f(b)==0.
        retval_r*=dr

        if (iu==0):
            # collect f(u=a) + f(u=b)
            int_r_u0=retval_r
        elif (iu==1):
            int_r_u1=retval_r
        retval += retval_r
    retval -= 0.5*(int_r_u0+int_r_u1)
    retval *= du
    return 16*M_PI**2*retval**2

cdef double F2GaussianEllipsoidScalar(double q, double a, double b,
                                      double rhocore, double rhoshell,
                                      double sigmain, double sigmaout,
                                      Py_ssize_t Nintr=100, Py_ssize_t Nintu=100, Py_ssize_t NintU=100) nogil:
    """Calculate the SPHERICALLY AVERAGED scattering intensity of an electron density 
    of an ellipsoidal gaussian shell. Everything is in a spherical polar coordinate 
    system with the z axis parallel with the axis of revolution of the ellipsoid.
    
    Inputs:
        q: magnitude of the scattering variable
        a: half axis of the ellipsoid of revolution, parallel to the axis of revolution (z)
        b: half axis of the ellipsoid of revolution, orthogonal to the axis of revolution (z)
        rhocore: electron density at r=0
        rhoshell: highest electron density of the shell
        sigmain: HWHM of the inner half Gaussian
        sigmaout: HWHM of the outer half Gaussian
        
    Outputs:
        the spherically averaged scattered intensity at q
    """
    cdef:
        double retval = 0
        double U=0
        double dU=1./(NintU-1)
        Py_ssize_t iU=0
    for iU in range(0, NintU):
        U=iU*dU
        retval += F2(q, U, a, b, rhocore, rhoshell, sigmain, sigmaout,Nintr, Nintu)
    retval -= 0.5*(F2(q,0,a,b,rhocore,rhoshell, sigmain, sigmaout, Nintr, Nintu)+
                   F2(q,1,a,b,rhocore,rhoshell, sigmain, sigmaout, Nintr, Nintu))
    return retval*dU

def F2GaussianEllipsoid(np.ndarray[np.double_t, ndim=1] q not None,
                        double a, double b, double rhocore, double rhoshell,
                        double sigmain, double sigmaout,
                        Py_ssize_t Nintr=100, Py_ssize_t Nintu=100, Py_ssize_t NintU=100):
    """Calculate the SPHERICALLY AVERAGED scattering intensity of an electron density 
    of an ellipsoidal gaussian shell. Everything is in a spherical polar coordinate 
    system with the z axis parallel with the axis of revolution of the ellipsoid.
    
    Inputs:
        q: magnitude of the scattering variable
        a: half axis of the ellipsoid of revolution, parallel to the axis of revolution (z)
        b: half axis of the ellipsoid of revolution, orthogonal to the axis of revolution (z)
        rhocore: electron density at r=0
        rhoshell: highest electron density of the shell
        sigmain: HWHM of the inner half Gaussian
        sigmaout: HWHM of the outer half Gaussian
        
    Outputs: np.ndarray, size and shape as of q
        the spherically averaged scattered intensity in the points of q
    """
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
        my_out[i]=F2GaussianEllipsoidScalar(my_q[i],a,b,rhocore,rhoshell,sigmain,sigmaout,Nintr,Nintu,NintU)
#        with gil:
#            print('({})'.format(my_out[i]),end='',flush=True)
    out = np.empty_like(q, np.double)
    for i in range(N):
        out[i]=my_out[i]
    PyMem_Free(my_q)
    PyMem_Free(my_out)
    return out
#    cdef:
#        np.broadcast it = np.broadcast(q, out)
#    while np.PyArray_MultiIter_NOTDONE(it):
#        q_ = (<double*> np.PyArray_MultiIter_DATA(it, 0))[0]
#        (<double*> np.PyArray_MultiIter_DATA(it, 1))[0] = F2GaussianEllipsoidScalar(
#            q_, a, b, rhocore, rhoshell, sigmain, sigmaout, Nintr, Nintu, NintU)
#        print('.',end='',flush=True)
#        np.PyArray_MultiIter_NEXT(it)
#    return out

cdef double A1(double a, double b) nogil:
    cdef:
        double nu=b/a
        double tmp=0
    if nu>1:
        tmp=sqrt(nu*nu-1)
        return asinh(tmp)/tmp*b
    elif nu<1:
        tmp=sqrt(1-nu*nu)
        return asin(tmp)/tmp*b
    else: # nu == 1
        return b

cdef double A2(double a, double b) nogil:
    cdef:
        double nu=b/a
        double tmp=0
    if nu>1:
        tmp=sqrt(nu*nu-1)
        return atan(tmp)/tmp*b*b
    elif nu<1:
        tmp=sqrt(1-nu*nu)
        return log((2*tmp-nu*nu+2)/(nu*nu))/2/tmp*b*b
    else:
        return b*b

cdef double A3(double a, double b) nogil:
    return a*b*b

cdef double A4(double a, double b) nogil:
    cdef:
        double nu=b/a
        double tmp=0
    if nu>1:
        tmp = sqrt(nu*nu-1)
        return (nu*nu*tmp*atan(tmp)+nu*nu-1)/(2*nu**4-2*nu**2)*b**4
    elif nu<1:
        tmp=sqrt(1-nu*nu)
        return -(nu**2*tmp*log((2*tmp-nu**2+2)/(nu**2))-2*nu**2+2)/(4*nu**4-4*nu**2)*b**4
    else:
        return b**4

cdef double A5(double a, double b) nogil:
    cdef:
        double nu=b/a
    return (2*nu**2+1)/3*a**5*nu**2

cdef void _rhofromI0Rg(double I0, double Rg, double a, double b, double sigmain, double sigmaout,
                      double *rhocore, double *rhoshell):
    cdef:
        double a1,a2,a3,a4,a5
        double A, B, C, D, E
    a1=A1(a,b)
    a2=A2(a,b)
    a3=A3(a,b)
    a4=A4(a,b)
    a5=A5(a,b)
    A=a5/5-3*sqrt(M_PI/2)*sigmain**5+8*a1*sigmain**4-3*sqrt(2*M_PI)*a2*sigmain**3+4*a3*sigmain**2-sqrt(M_PI/2)*a4*sigmain
    B=3*sqrt(M_PI/2)*(sigmain**5+sigmaout**5)+8*a1*(sigmaout**4-sigmain**4)+3*sqrt(2*M_PI)*a2*(sigmain**3+sigmaout**3) +4*a3*(sigmaout**2-sigmain**2)+sqrt(M_PI/2)*a4*(sigmain+sigmaout)
    C=a3/3-sqrt(M_PI/2)*sigmain**3+2*a1*sigmain**2-sqrt(M_PI/2)*a2*sigmain
    D=sqrt(M_PI/2)*(sigmaout**3+sigmain**3)+2*a1*(sigmaout**2-sigmain**2)+sqrt(M_PI/2)*a2*(sigmain+sigmaout)
    E = (B-Rg**2*D)/(C*Rg**2-A)
    rhoshell[0] = sqrt(I0)/(4*M_PI)/abs(E*C+D)
    rhocore[0] = rhoshell[0]*E

cdef void _I0Rgfromrho(double *I0, double *Rg, double a, double b, double sigmain, double sigmaout,
                      double rhocore, double rhoshell):
    cdef:
        double a1,a2,a3,a4,a5
        double A, B, C, D, E
    a1=A1(a,b)
    a2=A2(a,b)
    a3=A3(a,b)
    a4=A4(a,b)
    a5=A5(a,b)
    A=a5/5-3*sqrt(M_PI/2)*sigmain**5+8*a1*sigmain**4-3*sqrt(2*M_PI)*a2*sigmain**3+4*a3*sigmain**2-sqrt(M_PI/2)*a4*sigmain
    B=3*sqrt(M_PI/2)*(sigmain**5+sigmaout**5)+8*a1*(sigmaout**4-sigmain**4)+3*sqrt(2*M_PI)*a2*(sigmain**3+sigmaout**3) +4*a3*(sigmaout**2-sigmain**2)+sqrt(M_PI/2)*a4*(sigmain+sigmaout)
    C=a3/3-sqrt(M_PI/2)*sigmain**3+2*a1*sigmain**2-sqrt(M_PI/2)*a2*sigmain
    D=sqrt(M_PI/2)*(sigmaout**3+sigmain**3)+2*a1*(sigmaout**2-sigmain**2)+sqrt(M_PI/2)*a2*(sigmain+sigmaout)
    Rg[0]=sqrt((A*rhocore+B*rhoshell)/(C*rhocore+D*rhoshell))
    I0[0]=16*M_PI*M_PI*(C*rhocore+D*rhoshell)**2

def I0Rgfromrho(double a, double b, double rhocore, double rhoshell, double sigmain, double sigmaout):
    cdef double I0, Rg
    _I0Rgfromrho(&I0, &Rg, a,b,sigmain, sigmaout,rhocore, rhoshell)
    return (I0, Rg)

def rhofromI0Rg(double I0, double Rg, double a, double b, double sigmain, double sigmaout):
    cdef double rhocore, rhoshell
    _rhofromI0Rg(I0, Rg, a, b, sigmain, sigmaout, &rhocore, &rhoshell)
    return (rhocore, rhoshell)
