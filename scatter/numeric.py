from numpy.linalg import norm
from scatter.analytics import potential
from scatter.transform import getPropCoords
import numpy as np
import scatter.constants as c


def derivative(func, x, h=1e-8):
    return 0.5*(func(x+h)-func(x-h))/h


def numericDerivatives(r, R):
    rmag = norm(r)
    Rmag = norm(R)
    cosGamma = (r @ R)/(rmag * Rmag)

    dVdr = derivative(lambda x: potential(x, Rmag, cosGamma), rmag)
    dVdR = derivative(lambda x: potential(rmag, x, cosGamma), Rmag)
    dVdcosGamma = derivative(lambda x: potential(rmag, Rmag, x), cosGamma)
    dcosGammadr = (rmag/Rmag*R - cosGamma*r)/(rmag*rmag)
    dcosGammadR = (Rmag/rmag*r - cosGamma*R)/(Rmag*Rmag)
    pdot = -dVdr*r/rmag - dVdcosGamma*dcosGammadr
    Pdot = -dVdR*R/Rmag - dVdcosGamma*dcosGammadR

    print("--------------------------")
    print(dVdr, dVdcosGamma, dcosGammadr)
    print(dVdR, dVdcosGamma, dcosGammadR)

    return pdot, Pdot


def equation_of_motion(t, coordinates):
    # current jacobi coordinates
    r, p, R, P = getPropCoords(coordinates)

    # first derivatives w.r.t. t
    rdot = p/c.mu
    Rdot = P/c.M

    pdot, Pdot = numericDerivatives(r, R)

    return np.concatenate((rdot, pdot, Rdot, Pdot), axis=0)
