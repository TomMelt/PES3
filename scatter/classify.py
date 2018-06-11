from . import constants as c
from .numeric import diatomPEC
from .transform import internuclear
import numpy as np


def assignClassical(r, p, R, P):

    # relative distances of anti-proton w.r.t. proton
    R1, R2, R3 = internuclear(r, R)

    Rmin = np.min([R1, R2, R3])

    assignment = {}
    mu = 0.
    Ec = 0.

    if R1 == Rmin:
        mu = c.mu
        r_rel = r
        v_rel = p/c.mu
        assignment["case"] = 3
        Ec = 0.5*mu*v_rel@v_rel + diatomPEC(R1)
    if R2 == Rmin:
        mu = c.m1*c.m3/(c.m1 + c.m3)
        r_rel = R + c.mu/c.m1*r
        v_rel = P/c.MU + p/c.m1
        assignment["case"] = 1
        Ec = 0.5*mu*v_rel@v_rel - 1./R2
    if R3 == Rmin:
        mu = c.m2*c.m3/(c.m2 + c.m3)
        r_rel = R - c.mu/c.m2*r
        v_rel = P/c.MU - p/c.m2
        assignment["case"] = 1
        Ec = 0.5*mu*v_rel@v_rel - 1./R3

    if Ec > 0.:
        assignment["Ec"] = np.NAN
        assignment["lc"] = np.NAN
        assignment["case"] = 2
    else:
        assignment["Ec"] = Ec
        assignment["lc"] = c.mu*np.linalg.norm(np.cross(r_rel, v_rel))

    return assignment


def assignQuantum(Ec, lc):

    if Ec is np.NAN:
        return np.NAN, np.NAN

    # NOT GENERAL
    # need to make this general for any masses m1, m2 and m3
    nc = np.sqrt(c.mu/(2.*np.abs(Ec)))
    nq = 0
    for n in range(1, 1000):
        nl = np.power((n-1)*(n-0.5)*n, 1./3.)
        nh = np.power((n+1)*(n+0.5)*n, 1./3.)
        if nc >= nl and nc < nh:
            nq = n
            break

    lq = 0
    for l in range(1000):
        ll = l
        lh = (l+1)
        if nq/nc*lc > ll and nq/nc*lc <= lh:
            lq = l
            break

    return nq, lq
