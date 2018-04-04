from numpy import cos, sin
import numpy as np
import scatter.constants as c


def getPropCoords(coordinates):
    # get propagation coordinates

    r = coordinates[:3]
    p = coordinates[3:6]
    R = coordinates[6:9]
    P = coordinates[9:]

    return r, p, R, P


def getParticleCoords(r, p, R, P):
    # get lab frame coordinates

    r1 = c.mu/c.m2*r
    r2 = -c.mu/c.m1*r
    r3 = R

    return r1, r2, r3


def sphericalToCart(coordinates):
    # spherical r, theta and phi into cartesian x, y and z

    r = coordinates[0]
    t = coordinates[1]
    p = coordinates[2]

    x = r*cos(t)*sin(p)
    y = r*sin(t)*sin(p)
    z = r*cos(p)

    return np.array([x, y, z])
