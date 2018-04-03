import numpy as np
from numpy import sin, cos, tan, exp
from scipy.special import legendre
from scatter.transformation import getPropCoords


# Constants for the particles
m1 = 1.
m2 = 1.
mu = m1*m2/(m1+m2)
M = 1.


# Constants for the potential energy surface
alpha = 2.027
beta = 0.375
C = 17.283


def potential(r, R, cosGamma):
    # calculate the potential
    p2 = legendre(2)
    V = C*exp(-alpha*R)*(1.+beta*p2(cosGamma))
    return V


def Hamiltonian(r, t, p, R, T, P, pr, pt, pp, pR, pT, pP):
    # calculate the hamiltonian of the scattering system
    KE11 = pr*pr/(2.*mu)
    if p == 0.:
        KE12 = 0.
    else:
        KE12 = pt*pt/(2.*mu*r*r*sin(p)*sin(p))
    KE13 = pp*pp/(2.*mu*r*r)

    KE21 = pR*pR/(2.*M)
    if P == 0.:
        KE22 = 0.
    else:
        KE22 = pT*pT/(2.*M*R*R*sin(P)*sin(P))
    KE23 = pP*pP/(2.*M*R*R)

    cosGamma = sin(p)*sin(P)*cos(t-T) + cos(p)*cos(P)

    KE1 = KE11 + KE12 + KE13
    KE2 = KE21 + KE22 + KE23

    return KE1 + KE2 + potential(r, R, cosGamma)


def analyticDerivatives(
        r, t, p, R, T, P,
        pr, pt, pp, pR, pT, pP
        ):
    # calculate and return the analytic derivatives

    cosGamma = sin(p)*sin(P)*cos(t-T) + cos(p)*cos(P)
    dVdcosGamma = 3.*beta*C*exp(-alpha*R)*cosGamma
    dVdr = 0.
    dVdR = -alpha*potential(r, R, cosGamma)
    dcosGammadt = -sin(p)*sin(P)*sin(t-T)
    dcosGammadp = cos(p)*sin(P)*cos(t-T) - sin(p)*cos(P)
    dcosGammadT = -dcosGammadt
    dcosGammadP = sin(p)*cos(P)*cos(t-T) - cos(p)*sin(P)

    dr = 0.
    if p == 0.:
        dt = 0.
    else:
        dt = pt/(mu*r*r*sin(p)*sin(p))
    dp = pp/(mu*r*r)

    dR = pR/M
    if P == 0.:
        dT = 0.
    else:
        dT = pT/(M*R*R*sin(P)*sin(P))
    dP = pP/(M*R*R)

    dpr = pt/r*dt + pp/r*dp - dVdr
    dpt = -dVdcosGamma*dcosGammadt
    if p == 0.:
        dpp = -dVdcosGamma*dcosGammadp
    else:
        dpp = pt/(2.*tan(p))*dt - dVdcosGamma*dcosGammadp

    dpR = pT/R*dT + pP/R*dP - dVdR
    dpT = -dVdcosGamma*dcosGammadT
    if P == 0.:
        dpP = -dVdcosGamma*dcosGammadP
    else:
        dpP = pT/(2.*tan(P))*dT - dVdcosGamma*dcosGammadP

    return dr, dt, dp, dR, dT, dP, dpr, dpt, dpp, dpR, dpT, dpP


def equation_of_motion(t, coordinates, numeric=True):
    # current jacobi coordinates
    r, t, p, R, T, P, pr, pt, pp, pR, pT, pP = getPropCoords(coordinates)

    if numeric:
        return
    else:
        derivs = list(
                analyticDerivatives(
                    r, t, p, R, T, P,
                    pr, pt, pp, pR, pT, pP
                    )
                )
        return np.array(derivs)
