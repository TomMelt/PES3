from numpy import cos, sin
from numpy.linalg import norm
from . import constants as c
import numpy as np


def getPropCoords(coordinates):
    """Return a tuple containing the Jacobi coordinates (r, p, R, P) given an
    np.array containing all four coordinates concatenated together.
    coordinates -- current jacobi coordinates r, p, R and P concatenated as one
                   np.array.
    """
    if len(coordinates) != c.dim:
        raise IndexError(
                "coordinates is len={0} but it should be len={1}".format(
                    len(coordinates),
                    c.dim,
                    )
                )

    r = coordinates[:3]
    p = coordinates[3:6]
    R = coordinates[6:9]
    P = coordinates[9:]
    return r, p, R, P


def getParticleCoords(r, p, R, P):
    """Return a tuple containing spatial coords relative to the diatomic C.o.M
    and momenta relative to the system C.o.M (r1, r2, r3, p1, p2, p3).
    r -- C.o.M position for diatom (BC)
    p -- C.o.M conjugate momentum for diatom (BC)
    R -- C.o.M position for scattering particle and diatom (BC)
    P -- C.o.M conjugate momentum for scattering particle and diatom (BC)
    """

    # centre of mass coordinates
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

    r1 = -c.m3/c.Mt*R - c.mu/c.m1*r
    r2 = -c.m3/c.Mt*R + c.mu/c.m2*r
#    r1 = -c.mu/c.m1*r
#    r2 = c.mu/c.m2*r
    r3 = c.m3/c.Mt*R + c.MU/c.m3*R
    p1 = -c.mu/c.m2*P - p
    p2 = -c.mu/c.m1*P + p
    p3 = P
    return r1, r2, r3, p1, p2, p3


def sphericalToCart(r, theta, phi):
    """Return cartesian x, y and z as an np.array([x, y, z]) given the
    spherical coordinates r, theta and phi.
    r     -- radius [0, inf]
    theta -- azimuthal angle [0, 2Pi]
    phi   -- polar angle [0, Pi]
    """
    if r < 0.:
        msg = "r must be greater than or equal to zero"
        msg += "\nr = {0}"
        raise ValueError(msg.format(r))
    if theta > 2.*np.pi or theta < 0.:
        msg = "theta must be: 0 <= theta <= 2*Pi"
        msg += "\ntheta = {0}"
        raise ValueError(msg.format(theta))
    if phi > np.pi or phi < 0.:
        msg = "phi must be: 0 <= phi <= Pi"
        msg += "\nphi = {0}"
        raise ValueError(msg.format(phi))
    if any(isinstance(x, float) is False for x in [r, theta, phi]):
        msg = "r, theta and phi must be floats\n"
        msg += "r={0}\n"
        msg += "theta={1}\n"
        msg += "phi={2}"
        raise TypeError(
                msg.format(
                    type(r),
                    type(theta),
                    type(phi),
                    )
                )

    x = r*cos(theta)*sin(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(phi)
    return np.array([x, y, z])


def perpMomentum(magnitude, theta, phi, eta):
    """Return the momenutm vector that is perpendicular to the molecular axis
    (z-direction in this case)
    magnitude -- magnitude of vector
    theta     -- azimuthal angle [0, 2Pi]
    phi       -- polar angle [0, Pi]
    eta       -- arbitrary angle [0, 2Pi]
    """

    px = -magnitude*(sin(theta)*cos(eta) + cos(theta)*cos(phi)*sin(eta))
    py = magnitude*(cos(theta)*cos(eta) - sin(theta)*cos(phi)*sin(eta))
    pz = magnitude*(sin(phi)*sin(eta))

    return np.array([px, py, pz])


def internuclear(r, R):
    """Return the internuclear distances as a tuple (R1, R2, R3).
    r -- C.o.M position for diatom (BC)
    R -- C.o.M position for scattering particle and diatom (BC)
    """
    if any(len(x) != 3 for x in [r, R]):
        msg = "arrays must be 3 dimensional"
        msg += "\nr = {0}"
        msg += "\nR = {1}"
        raise IndexError(msg.format(r, R))
    if any(isinstance(x, np.ndarray) is False for x in [r, R]):
        msg = "r and R must be np.arrays\n"
        msg += "r={0}\n"
        msg += "R={1}"
        raise TypeError(
                msg.format(
                    type(r),
                    type(R),
                    )
                )

    R1 = norm(r)
    R2 = norm(c.mu/c.m1*r + R)
    R3 = norm(c.mu/c.m2*r - R)
    return R1, R2, R3
