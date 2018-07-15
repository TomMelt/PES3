from .transform import getPropCoords, internuclear
from . import constants as c
import numpy as np
from . import pes


def pesWrapper(R1, R2, R3):
    """A wrapper function around the pes.so object. It also converts from mEh
    to Eh. As well as ensuring the coordinates are parsed in the correct order.
    Returns the value of the potential energy surface at the point R1, R2, R3.
    R1 -- distance between BC (diatom)
    R2 -- distance between AB
    R3 -- distance between AC
    """
    if any(x < 0. for x in [R1, R2, R3]):
        msg = "R1, R2 and R3 must be postive floats.\n"
        msg += "R1={0}\n"
        msg += "R2={1}\n"
        msg += "R3={2}"
        raise ValueError(
                msg.format(
                    R1,
                    R2,
                    R3,
                    )
                )

    if any(isinstance(x, float) is False for x in [R1, R2, R3]):
        msg = "R1, R2 and R3 must be floats.\n"
        msg += "R1={0}\n"
        msg += "R2={1}\n"
        msg += "R3={2}"
        raise TypeError(
                msg.format(
                    type(R1),
                    type(R2),
                    type(R3),
                    )
                )

    return pes.pes3(R2, R3, R1)/1000.


def diatomPEC(R1):
    """diatomic PEC for molecule BC
    R1 -- distance between BC (diatom)
    """
    if R1 < 0.:
        msg = "R1 must be a postive float.\n"
        msg += "R1={0}"
        raise ValueError(
                msg.format(
                    R1,
                    )
                )

    if isinstance(R1, float) is False:
        msg = "R1 must be a float.\n"
        msg += "R1={0}"
        raise TypeError(
                msg.format(
                    type(R1),
                    )
                )

    R = 1000.
    return pesWrapper(R1, R, R)


def potential(R1, R2, R3):
    """Interaction potential as a function of the three internuclear distances
    R1 -- distance between BC (diatom)
    R2 -- distance between AB
    R3 -- distance between AC
    """
    if any(x < 0. for x in [R1, R2, R3]):
        msg = "R1, R2 and R3 must be postive floats.\n"
        msg += "R1={0}\n"
        msg += "R2={1}\n"
        msg += "R3={2}"
        raise ValueError(
                msg.format(
                    R1,
                    R2,
                    R3,
                    )
                )

    if any(isinstance(x, float) is False for x in [R1, R2, R3]):
        msg = "R1, R2 and R3 must be floats.\n"
        msg += "R1={0}\n"
        msg += "R2={1}\n"
        msg += "R3={2}"
        raise TypeError(
                msg.format(
                    type(R1),
                    type(R2),
                    type(R3),
                    )
                )

    V = pesWrapper(R1, R2, R3)
    return V


def Hamiltonian(r, p, R, P):
    """Return the total energy of the system
    r -- C.o.M position for diatom (BC)
    p -- C.o.M conjugate momentum for diatom (BC)
    R -- C.o.M position for scattering particle and diatom (BC)
    P -- C.o.M conjugate momentum for scattering particle and diatom (BC)
    """
    for vector in [r, p, R, P]:
        if len(vector) == 3:
            continue
        else:
            msg = "input vectors must have exactly 3 dimensions"
            msg += "\n"
            msg += "r={0}\n"
            msg += "p={1}\n"
            msg += "R={2}\n"
            msg += "P={3}"
            raise IndexError(msg.format(r, p, R, P))

    R1, R2, R3 = internuclear(r, R)
    H = p@p/(2.*c.mu) + P@P/(2.*c.MU) + potential(R1, R2, R3)
    return H


def Lagrangian(r, p, R, P):
    """Return the Lagrangian of the system
    r -- C.o.M position for diatom (BC)
    p -- C.o.M conjugate momentum for diatom (BC)
    R -- C.o.M position for scattering particle and diatom (BC)
    P -- C.o.M conjugate momentum for scattering particle and diatom (BC)
    """

    R1, R2, R3 = internuclear(r, R)
    L = p@p/(2.*c.mu) + P@P/(2.*c.MU) - potential(R1, R2, R3)
    return L


def derivative(func, x, method="stencil", h=1e-06):
    """Compute numerical derivative of function "func" at position x
    func   -- function or lambda function of variable "x"
    x      -- position at which derivative is computed
    method -- "euler" or "stencil" (5 point)
    h      -- delta used in derivative (don't make too small i.e., < 1e-08)
    """

    if method.upper() == "EULER":
        return 0.5*(func(x+h)-func(x-h))/h
    if method.upper() == "STENCIL":
        temp = -func(x+2.*h) + 8.*func(x+h) - 8.*func(x-h) + func(x-2.*h)
        return temp/(12.*h)


def numericDerivatives(r, R):
    """Calculate the numeric derivatives "pdot" and "Pdot" for the equations of
    motion. Returns a tuple containing the two np.arrays.
    r -- C.o.M position for diatom (BC)
    R -- C.o.M position for scattering particle and diatom (BC)
    """

    for vector in [r, R]:
        if len(vector) == 3:
            continue
        else:
            msg = "input vectors must have exactly 3 dimensions\n"
            msg += "r={0}\n"
            msg += "R={1}"
            raise IndexError(msg.format(r, R))

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
    """Return the equations of motion for the system.
    coordinates -- current jacobi coordinates r, p, R and P concatenated as one
                   np.array.
    t           -- time of current step
    """

    if not isinstance(t, float):
        raise TypeError(
                "time t must be a float. t={0}".format(type(t))
                )
    if len(coordinates) != c.dim:
        raise IndexError(
                "coordinates is len={0} but it should be len={1}".format(
                    len(coordinates),
                    c.dim,
                    )
                )

    r, p, R, P = getPropCoords(coordinates)

    # first derivatives w.r.t. t
    rdot = p/c.mu
    Rdot = P/c.MU

    pdot, Pdot = numericDerivatives(r, R)

    return np.concatenate((rdot, pdot, Rdot, Pdot), axis=0)
