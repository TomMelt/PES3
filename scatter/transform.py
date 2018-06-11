from numpy import cos, sin
from numpy.linalg import norm
from . import constants as c
import numpy as np


def getPropCoords(coordinates):
    # get propagation coordinates

    r = coordinates[:3]
    p = coordinates[3:6]
    R = coordinates[6:9]
    P = coordinates[9:]

    return r, p, R, P


def getParticleCoords(r, p, R, P):
    # spatial coords are relative to the diatomic C.o.M
    # momenta are relative to the system C.o.M

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


def sphericalToCart(r, t, p):
    # spherical r, theta and phi into cartesian x, y and z

    x = r*cos(t)*sin(p)
    y = r*sin(t)*sin(p)
    z = r*cos(p)

    return np.array([x, y, z])


def perpMomentum(m, t, p, n):

    px = -m*(sin(t)*cos(n) + cos(t)*cos(p)*sin(n))
    py = m*(cos(t)*cos(n) - sin(t)*cos(p)*sin(n))
    pz = m*(sin(p)*sin(n))

    return np.array([px, py, pz])


def internuclear(r, R):
    R1 = norm(r)
    R2 = norm(c.mu/c.m1*r + R)
    R3 = norm(c.mu/c.m2*r - R)
    return R1, R2, R3


def distanceFromCoM(ma, mb, ra, rb, rc):
    rcom = (ma*ra + mb*rb)/(ma + mb)
    dist = norm(rcom - rc)
    return dist
