from numpy.linalg import norm
from scatter.transform import getPropCoords
from scipy.special import legendre
import numpy as np
import scatter.constants as c


def potential(r, R, cosGamma):
    p2 = legendre(2)
    V = c.C*np.exp(-c.alpha*R)*(1.+c.beta*p2(cosGamma)) + 0.5*c.k*(r-c.re)*(r-c.re)
    return V


def analyticDerivatives(r, R):
    rmag = norm(r)
    Rmag = norm(R)
    cosGamma = (r @ R)/(rmag * Rmag)

    runit = r/rmag
    Runit = R/Rmag

    dVdr = c.k*(rmag-c.re)
    dVdR = -c.alpha*potential(rmag, Rmag, cosGamma)

    dVdcosGamma = 3.*c.beta*c.C*np.exp(-c.alpha*Rmag)*cosGamma
    dcosGammadr = (rmag/Rmag*R - cosGamma*r)/(rmag*rmag)
    dcosGammadR = (Rmag/rmag*r - cosGamma*R)/(Rmag*Rmag)

    pdot = -dVdr*runit - dVdcosGamma*dcosGammadr
    Pdot = -dVdR*Runit - dVdcosGamma*dcosGammadR

    return pdot, Pdot


def equation_of_motion(t, coordinates):
    # current jacobi coordinates
    r, p, R, P = getPropCoords(coordinates)

    # first derivatives w.r.t. t
    rdot = p/c.mu
    Rdot = P/c.M

    pdot, Pdot = analyticDerivatives(r, R)

    return np.concatenate((rdot, pdot, Rdot, Pdot), axis=0)
