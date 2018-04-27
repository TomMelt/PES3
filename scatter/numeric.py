from pk import PK as potential
from scatter.transform import getPropCoords, internuclear
import numpy as np
import pk
import scipy.optimize as opt
import scatter.constants as c
import transform
import random


def rovibrationalEnergy(v, J):
    E = 0.
    for i in range(3):
        E += c.G[i]*np.power(v+0.5, i+1)
        for j in range(3):
            E += c.F[i, j]*np.power(v+0.5, j)*np.power(J*(J+1.), i+1)
    return E


def rootFunctional(v, J, R):
    E = J*(J+1)/(2.*c.mu*R*R) + pk.D1*np.power(1.-np.exp(-pk.a*(R-pk.Re)), 2)
    return E - rovibrationalEnergy(v, J)


def initialiseDiatomic(v, J, rand):

    theta = rand.random()*2.*np.pi
    cosphi = -1. + 2.*rand.random()
    phi = np.arccos(cosphi)
    eta = rand.random()*2.*np.pi

    r_plus = opt.brentq(lambda x: rootFunctional(v, J, x), c.rmax, pk.Re)
    r_minus = opt.brentq(lambda x: rootFunctional(v, J, x), c.rmin, pk.Re)

    r = rand.choice([r_plus, r_minus])
    # print("r-:", r_minus, " r+:", r_plus, "r:", r)

    ri = transform.sphericalToCart(np.array([r, theta, phi]))

    p = np.sqrt(J*(J+1.))/r

    pi = transform.perpMomentum(p, theta, phi, eta)

    return ri, pi


def initialiseScattering(rand, epsilon):

    b = np.sqrt(rand.random()*c.bmax*c.bmax)

    X = b
    Y = 0.
    Z = np.sqrt(c.R0*c.R0 - b*b)
    Ri = np.array([X, Y, Z])

    PX = 0.
    PY = 0.
    PZ = np.sqrt(2.*c.MU*epsilon)
    Pi = -np.array([PX, PY, PZ])

    return Ri, Pi


def derivative(func, x, h=1e-8):
    return 0.5*(func(x+h)-func(x-h))/h


def numericDerivatives(r, R):
    R1, R2, R3 = internuclear(r, R)

    dVdR1 = derivative(lambda x: potential(x, R2, R3), R1)
    dVdR2 = derivative(lambda x: potential(R1, x, R3), R2)
    dVdR3 = derivative(lambda x: potential(R1, R2, x), R3)

    dR1dr = 1./R1*r
    dR2dr = c.mu/(c.m1*R2)*(c.mu/c.m1*r + R)
    dR3dr = c.mu/(c.m2*R3)*(c.mu/c.m2*r - R)

    dR1dR = np.array([0., 0., 0.])
    dR2dR = c.m1/c.mu*dR2dr
    dR3dR = -c.m2/c.mu*dR3dr

    pdot = -dVdR1*dR1dr - dVdR2*dR2dr - dVdR3*dR3dr
    Pdot = -dVdR1*dR1dR - dVdR2*dR2dR - dVdR3*dR3dR

#    print("--------------------------")
#    print(dVdR1, dVdR2, dVdR3)
#    print(dR1dr, dR2dr, dR3dr)
#    print(dR1dR, dR2dR, dR3dR)

    return pdot, Pdot


def equation_of_motion(t, coordinates):
    # current jacobi coordinates
    r, p, R, P = getPropCoords(coordinates)

    # first derivatives w.r.t. t
    rdot = p/c.mu
    Rdot = P/c.MU

    pdot, Pdot = numericDerivatives(r, R)

    return np.concatenate((rdot, pdot, Rdot, Pdot), axis=0)
