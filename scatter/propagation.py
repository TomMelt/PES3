from transform import distanceFromCoM, getParticleCoords
from transform import getPropCoords, internuclear
import constants as c
from initialize import getInitialConditions
import numeric as num
import numpy as np
import scipy.integrate as odeint


# TODO:
# - add quantum classification after end of each trajectory
# - add/fix tests
# - bin minimum distance between atoms??

# NOTE:
# ----------------------------------------------------------------------------
# - initially program was also built in spherical coords but this does not
#   work well with RK4 prop because angles can be arbitrarily large
#   and there are singularities in the equations of motion (1/sin^2(phi)) etc.
# ----------------------------------------------------------------------------
# - "one of the necessary conditions for the proper description in terms of a
#   classical traj. is that changes in the De Broglie wavelength of the
#   appropriate vars. be small over the scale determined by the spatial
#   variation of the wavefunction"
#   And "epsilon*b*b = l_c^2/2Mu"
#   E.E.Nikitin
#   Jost W. (Ed.), Phys. chem., 6a, Academic Press, New York (1974)
#   ch. 4
# ----------------------------------------------------------------------------
def runTrajectory(seed, trajID, v, J, epsilon, returnTraj=False):
    ri, pi, Ri, Pi = getInitialConditions(seed, trajID, v, J, epsilon)
    return propagate(ri, pi, Ri, Pi, returnTraj)


def isConverged(dist):
    """Return True if trajectory is converged
    dist -- distance between particle and C.o.M of the other two
    """
    c1 = dist > c.R0
    return c1


def propagate(ri, pi, Ri, Pi, returnTraj=False):
    """Propagate trajectory from a given intial condition based on Hamilton's
    Equations.
    returnTraj -- flag to return trajectory for plotting (default False)
    """

    initialConditions = np.concatenate([ri, pi, Ri, Pi], axis=0)

    # initialise stepping object
    stepper = odeint.RK45(
            lambda t, y: num.equation_of_motion(t, y),
            c.ts, initialConditions, c.tf,
            max_step=c.maxstep, rtol=c.rtol, atol=c.atol
            )

    r, p, R, P = getPropCoords(stepper.y)
    H = num.Hamiltonian(r, p, R, P)

    # array to store trajectories
    trajectory = []
    trajectory.append([stepper.t] + stepper.y.tolist() + [H])
    dist = 0.
    maxstep, maxErr = 0., 0.
    countstep = 0

    # propragate Hamilton's eqn's
    while stepper.t < c.tf:
        try:

            stepper.step()

            r, p, R, P = getPropCoords(stepper.y)
            R1, R2, R3 = internuclear(r, R)
            r1, r2, r3, p1, p2, p3 = getParticleCoords(r, p, R, P)

            if R1 == np.max([R1, R2, R3]):
                dist = distanceFromCoM(c.m2, c.m3, r2, r3, r1)
            if R2 == np.max([R1, R2, R3]):
                dist = distanceFromCoM(c.m1, c.m3, r1, r3, r2)
            if R3 == np.max([R1, R2, R3]):
                dist = distanceFromCoM(c.m1, c.m2, r1, r2, r3)

            trajectory.append([stepper.t] + stepper.y.tolist() + [H])

            maxstep = np.max([stepper.step_size, maxstep])
            maxErr = np.max(
                    [np.abs((H - num.Hamiltonian(r, p, R, P)/H)), maxErr]
                    )
            countstep = countstep + 1

            if isConverged(dist):
                tf = stepper.t
                break

            H = num.Hamiltonian(r, p, R, P)
        except RuntimeError as e:
            print(e)
            break
    trajectory = np.array(trajectory)

    if returnTraj:
        return trajectory
    else:
        return tuple(ri) + tuple(pi) + tuple(Ri) + tuple(Pi) + tuple(r) + tuple(p) + tuple(R) + tuple(P) + tuple([tf, maxErr, countstep, maxstep])
        # return ri, pi, Ri, Pi, r, p, R, P, tf, maxErr, countstep, maxstep
