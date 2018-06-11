from .transform import getPropCoords, internuclear
from . import constants as c
import numpy as np
from . import pes


@np.vectorize
def pesWrapper(R1, R2, R3):
    """Fortran function pes.pes3 needs conversion from mEh to Eh
    R1 -- distance between BC (diatom)
    R2 -- distance between AB
    R3 -- distance between AC
    """
    return pes.pes3(R2, R3, R1)/1000.


def diatomPEC(R1):
    """diatomic PEC for molecule BC
    R1 -- distance between BC (diatom)
    """
    R = 1000.
    return pesWrapper(R1, R, R)


def potential(R1, R2, R3):
    """Interaction potential as a function of the three internuclear distances
    R1 -- distance between BC (diatom)
    R2 -- distance between AB
    R3 -- distance between AC
    """
    V = pesWrapper(R1, R2, R3)
    return V


def Hamiltonian(r, p, R, P):
    """Calculate the total energy of the system
    r -- C.o.M position for diatom (BC)
    p -- C.o.M conjugate momentum for diatom (BC)
    R -- C.o.M position for scattering particle and diatom (BC)
    P -- C.o.M conjugate momentum for scattering particle and diatom (BC)
    """

    R1, R2, R3 = internuclear(r, R)
    H = p@p/(2.*c.mu) + P@P/(2.*c.MU) + potential(R1, R2, R3)
    return H


def derivative(func, x, method="stencil", h="1e-06"):
    """Compute numerical derivative of function "func"
    func -- function or lambda function of variable "x"
    x -- position at which derivative is computed
    method -- "euler" or "stencil" (5 point)
    h -- delta used in derivative (don't make too small i.e., < 1e-08)
    """

    if method.upper() == "EULER":
        return 0.5*(func(x+h)-func(x-h))/h
    if method.upper() == "STENCIL":
        temp = -func(x+2.*h) + 8.*func(x+h) - 8.*func(x-h) + func(x-2.*h)
        return temp/(12.*h)


def numericDerivatives(r, R):
    R1, R2, R3 = internuclear(r, R)

    dVdR1 = derivative(
            lambda x: potential(x, R2, R3),
            R1,
            method=c.dmethod,
            h=c.dtol)
    dVdR2 = derivative(
            lambda x: potential(R1, x, R3),
            R2,
            method=c.dmethod,
            h=c.dtol)
    dVdR3 = derivative(
            lambda x: potential(R1, R2, x),
            R3,
            method=c.dmethod,
            h=c.dtol)

    dR1dr = 1./R1*r
    dR2dr = c.mu/(c.m1*R2)*(c.mu/c.m1*r + R)
    dR3dr = c.mu/(c.m2*R3)*(c.mu/c.m2*r - R)

    dR1dR = np.array([0., 0., 0.])
    dR2dR = c.m1/c.mu*dR2dr
    dR3dR = -c.m2/c.mu*dR3dr

    pdot = -dVdR1*dR1dr - dVdR2*dR2dr - dVdR3*dR3dr
    Pdot = -dVdR1*dR1dR - dVdR2*dR2dR - dVdR3*dR3dR

    return pdot, Pdot


def equation_of_motion(t, coordinates):
    # current jacobi coordinates
    r, p, R, P = getPropCoords(coordinates)

    # first derivatives w.r.t. t
    rdot = p/c.mu
    Rdot = P/c.MU

    pdot, Pdot = numericDerivatives(r, R)

    return np.concatenate((rdot, pdot, Rdot, Pdot), axis=0)
