from .numeric import diatomPEC
from . import constants as c
import numpy as np
import random as rnd
import scipy.integrate as integrate
import scipy.optimize as opt
from . import transform


def set_rand(seed, trajID):
    """Return num=c.numRandom random numbers. Each time a random number is
    required it can be obtained from the list of length num using .pop().
    This is to protect against the case where two independent trajectories,
    i.e., with different trajID's, use the same random numbers.
    seed -- any positive integer (see python random for more info)
    trajID -- any positive integer (limited only by python random)
    """
    rand = rnd
    rand.seed(seed)
    for i in range(trajID*c.numRandom):
        rand.random()
    nums = [rand.random() for x in range(c.numRandom)]
    return nums


def roots(v, J):
    """Return the internuclear turning points where the energy of the initial
    diatom in state (v, J) is equal to the diatomic PEC.
    v -- vibrational quantum number
    J -- rotational quantum number
    """
    r_plus = opt.brentq(lambda x: rootFunctional(v, J, x), c.rmax, c.Re)
    r_minus = opt.brentq(lambda x: rootFunctional(v, J, x), c.rmin, c.Re)
    return r_plus, r_minus


def vibrationalPeriod(v, J):
    """calculate vibrational period of diatom in initial state (v, J)
    using quadrature
    see p. 513 Truhlar D.G., Muckerman J.T. (1979)
    "Reactive Scattering Cross Sections III: Quasiclassical and
    Semiclassical Methods"
    v -- vibrational quantum number
    J -- rotational quantum number
    """
    r_plus, r_minus = roots(v, J)
    integral, error = integrate.quad(
            lambda x: 1./np.sqrt(
                rovibrationalEnergy(v, J)
                - (diatomPEC(x) - diatomPEC(2.))
                - J*(J + 1.)/(2.*c.mu*x*x)
                ),
            r_minus,
            r_plus
            )
    t = np.sqrt(2.*c.mu)*integral
    return t


def rovibrationalEnergy(v, J):
    """return rovibrational energy for a given initial state
    currently this is simply hard-coded in the constants module.
    Really the fortran code should be used for more general application
    v -- vibrational quantum number
    J -- rotational quantum number
    """
    E = c.rovib[J, v]
    return E


def rootFunctional(v, J, R):
    """This is the function that must be solved iteratively to find the
    turning points r+ and r-
    v -- vibrational quantum number
    J -- rotational quantum number
    R -- internuclear bond separation
    """
    E = J*(J+1)/(2.*c.mu*R*R) + diatomPEC(R) - diatomPEC(2.)
    return E - rovibrationalEnergy(v, J)


def initialiseDiatomic(v, J, rand):
    """return initial conditions for a diatom for a given state v, J
    v -- vibrational quantum number
    J -- rotational quantum number
    rand -- list of random numbers
    """

    theta = rand.pop()*2.*np.pi
    cosphi = rand.pop()
    phi = np.arccos(cosphi)
    eta = rand.pop()*2.*np.pi

    r_plus, r_minus = roots(v, J)

    # always pick the inner turning point (r_minus) because
    # the variation in initial separation (r) is handled via
    # the vibrational phase in the initial condition of the
    # scattering particle
    r = r_minus
    ri = transform.sphericalToCart(r, theta, phi)

    p = np.sqrt(J*(J+1.))/r
    pi = transform.perpMomentum(p, theta, phi, eta)

    return ri, pi


def initialiseScattering(rand, v, J, epsilon, bmax):
    """return initial conditions for the scattering particle
    rand -- random number generator
    epsilon -- scattering energy in the C.o.M (of the system)
    """

    PX = 0.
    PY = 0.
    PZ = -np.sqrt(2.*c.MU*epsilon)
    Pi = np.array([PX, PY, PZ])

    # initial separation (inc. vibrational phase of diatom)
    tau = vibrationalPeriod(v, J)
    rho = c.R0 + rand.pop()*np.linalg.norm(Pi)*tau/c.MU

    # scattering parameter
    b = np.sqrt(rand.pop())*bmax

    X = b
    Y = 0.
    Z = np.sqrt(rho*rho - b*b)
    Ri = np.array([X, Y, Z])

    return Ri, Pi


def getInitialConditions(seed, trajID, v, J, epsilon, bmax):
    """return initial conditions for the system
    v -- vibrational quantum number
    J -- rotational quantum number
    epsilon -- scattering energy in the C.o.M (of the system)
    """

    rand = set_rand(seed, trajID)

    ri, pi = initialiseDiatomic(v, J, rand)
    Ri, Pi = initialiseScattering(rand, v, J, epsilon, bmax)

    return ri, pi, Ri, Pi
