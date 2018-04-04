from scatter.plot import plotKE, plotStats, plot3Dtrace # noqa
from scatter.transform import getPropCoords
import numpy as np
import random as rnd
import scatter.analytics as ana
import scatter.constants as c
# import scatter.numeric as num
import scipy.integrate as odeint
import sys
import transform
# import xarray as xr


# TODO:
# - add functionality to terminate program on convergence of cross-section
#   not just on the max number of trajectories
# - add quantum classification after end of each trajectory
# - save output data to file
# - add/fix tests
# - check analytic vs numeric results
# - change masses for He-H2 (currently all 1 a.u.)
# - start separate analysis code?

# NOTE:
# - initially program was also built in spherical coords but this does not
#   work well with RK4 prop because angles can be arbitrarily large
#   and there are singularities in the equations of motion (1/sin^2(phi)) etc.


def initialiseRotor(rand):

    r = c.re
    theta = rand.random()*2.*np.pi
    phi = rand.random()*np.pi

    ri = transform.sphericalToCart(np.array([r, theta, phi]))
    pi = np.array([0., 0., 0.])

    return ri, pi


def initialiseScattering(rand, epsilon):

    b = rand.random()*c.bmax

    X = b
    Y = 0.
    Z = np.sqrt(c.R0*c.R0 - b*b)
    Ri = np.array([X, Y, Z])

    PX = 0.
    PY = 0.
    PZ = np.sqrt(2.*c.MU*epsilon)
    Pi = -np.array([PX, PY, PZ])

    return Ri, Pi


def isConverged(KE1a, KE1b, R):
    c1 = np.abs(KE1a - KE1b) < c.econ
    c2 = np.linalg.norm(R) > c.R0
    return c1 and c2


def main(args):

    seed = int(args[0])
    rand = rnd
    rand.seed(seed)

    # time range
    ts = 0.
    tf = 1000.

    epsilon = float(args[1])

    data = []

    countInelastic = 0
    countTotal = 0

    for i in range(500):

        # initial conditions of the scattering particles
        ri, pi = initialiseRotor(rand)
        Ri, Pi = initialiseScattering(rand, epsilon)
        b = Ri[0]
        initialConditions = np.concatenate((ri, pi, Ri, Pi), axis=0)

        # initialise stepping object
        stepper = odeint.RK45(
                lambda t, y: ana.equation_of_motion(t, y),
                ts, initialConditions, tf,
                max_step=c.maxstep, rtol=c.rtol, atol=c.atol
                )

        # array to store trajectories
        trajectory = []
        trajectory.append([stepper.t] + stepper.y.tolist())
        r, p, R, P = getPropCoords(stepper.y)
        KE1 = p@p/(2.*c.mu)
        KE2 = P@P/(2.*c.M)

        # propragate Hamilton's eqn's
        while stepper.t < tf:
            try:
                stepper.step()
                trajectory.append([stepper.t] + stepper.y.tolist())
#                print(stepper.step_size)
                r, p, R, P = getPropCoords(stepper.y)
                if isConverged(KE1, p@p/(2.*c.mu), R):
                    KE1 = p@p/(2.*c.mu)
                    KE2 = P@P/(2.*c.M)
                    countTotal += 1
                    if KE1 > 1e-3:
                        countInelastic += 1
                    break
                KE1 = p@p/(2.*c.mu)
                KE2 = P@P/(2.*c.M)
            except RuntimeError as e:
                print(e)
                break
        trajectory = np.array(trajectory)

        # plotKE(trajectory)
        # plot3Dtrace(trajectory)

        data.append([b, KE1, float(countInelastic)/float(countTotal)])

    print(epsilon, float(countInelastic)/float(countTotal))

    # plotStats(data)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
