import numpy as np
from numpy import sin, cos


m1 = 1.
m2 = 1.
mu = m1*m2/(m1+m2)
M = 1.


def getPropCoords(coordinates):
    # Get propagation coordinates (phase-space configuration)

    r = coordinates[0]
    t = coordinates[1]
    p = coordinates[2]
    R = coordinates[3]
    T = coordinates[4]
    P = coordinates[5]
    pr = coordinates[6]
    pt = coordinates[7]
    pp = coordinates[8]
    pR = coordinates[9]
    pT = coordinates[10]
    pP = coordinates[11]

    return r, t, p, R, T, P, pr, pt, pp, pR, pT, pP


def getParticleCoords(
        r, t, p, R, T, P,
        pr, pt, pp, pR, pT, pP
        ):
    # Get cartesian position of each particle
    r1 = m2/(m1+m2)*r*np.array([cos(t)*sin(p), sin(t)*sin(p), cos(p)])
    r2 = -m1/(m1+m2)*r*np.array([cos(t)*sin(p), sin(t)*sin(p), cos(p)])
    r3 = R*np.array([cos(T)*sin(P), sin(T)*sin(P), cos(P)])

    return r1, r2, r3
