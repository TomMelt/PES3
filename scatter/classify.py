from numeric import potential
from transform import internuclear
import constants as c
import numpy as np


def assignClassical(r, p, R, P):

    # relative distances of anti-proton w.r.t. proton
    R1, R2, R3 = internuclear(r, R)

    Rmin = np.min([R1, R2, R3])

    assignment = {}
    mu = 0.

    if R1 == Rmin:
        mu = c.mu
        r_rel = r
        v_rel = p/c.mu
        assignment["case"] = 3
    if R2 == Rmin:
        mu = c.m1*c.m3/(c.m1 + c.m3)
        r_rel = R + c.mu/c.m1*r
        v_rel = P/c.MU + p/c.m1
        assignment["case"] = 1
    if R3 == Rmin:
        mu = c.m2*c.m3/(c.m2 + c.m3)
        r_rel = R - c.mu/c.m2*r
        v_rel = P/c.MU - p/c.m2
        assignment["case"] = 1

    Ec = 0.5*mu*v_rel@v_rel + potential(R1, R2, R3)

#    print(R1, R2, R3, Ec)

    if Ec > 0.:
        assignment["Ec"] = np.NAN
        assignment["lc"] = np.NAN
        assignment["case"] = 2
    else:
        assignment["Ec"] = Ec
        assignment["lc"] = c.mu*np.linalg.norm(np.cross(r_rel, v_rel))

#    Ec2 = 0.5*mu1*v2@v2 - 1./d2
#    Ec3 = 0.5*mu2*v3@v3 - 1./d3

    return assignment


def assignQuantum(Ec, lc):

    if Ec is np.NAN:
        return np.NAN, np.NAN

    nc = np.sqrt(c.m3/(2.*np.abs(Ec)))
    nq = 0
    for n in range(1, 500):
        nl = np.power((n-1)*(n-0.5)*n, 1./3.)
        nh = np.power((n+1)*(n+0.5)*n, 1./3.)
        if nc >= nl and nc < nh:
            nq = n
            break

    lq = 0
    for l in range(500):
        ll = l
        lh = (l+1)
        if nq/nc*lc > ll and nq/nc*lc <= lh:
            lq = l
            break

    return nq, lq
