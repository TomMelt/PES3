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
    # r1 = -c.m3/c.Mt*R - c.mu/c.m1*r
    # r2 = -c.m3/c.Mt*R + c.mu/c.m2*r

    r1 = -c.mu/c.m1*r
    r2 = c.mu/c.m2*r
    r3 = c.m3/c.Mt*R + c.MU/c.m3*R
    p1 = -c.mu/c.m2*P - p
    p2 = -c.mu/c.m1*P + p
    p3 = P

    return r1, r2, r3, p1, p2, p3


def sphericalToCart(r, theta, phi):
    """Return cartesian x, y and z given the spherical coordinates r, theta and
    phi.
    r     -- radius [0, inf]
    theta -- azimuthal angle [0, 2Pi]
    phi   -- polar angle [0, Pi]
    """

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

    R1 = norm(r)
    R2 = norm(c.mu/c.m1*r + R)
    R3 = norm(c.mu/c.m2*r - R)
    return R1, R2, R3
