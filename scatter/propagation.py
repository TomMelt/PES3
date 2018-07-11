from .classify import assignQuantum, assignClassical
from .initialize import getInitialConditions
from .transform import getPropCoords, internuclear
from . import constants as c
from . import numeric as num
import numpy as np
import scipy.integrate as odeint
import xarray as xr


def runTrajectory(seed, trajID, v, J, epsilon, bmax, returnTraj=False):
    """Run a single trajectory and return the results as an xarray dataset.
    seed -- any positive integer (see python random for more info)
    trajID -- any positive integer (limited only by python random)
    v -- vibrational quantum number
    J -- rotational quantum number
    epsilon -- scattering energy in the C.o.M (of the system)
    bmax -- maximum value of the scattering parameter
    returnTraj -- set True to return the trajectory coordinates for each
                  time step as an array
    """

    ri, pi, Ri, Pi = getInitialConditions(seed, trajID, v, J, epsilon, bmax)

    if returnTraj:
        return propagate(ri, pi, Ri, Pi, returnTraj)

    result = propagate(ri, pi, Ri, Pi, returnTraj)

    if result['fail']:
        print(f'seed:{seed}, trajID:{trajID}, v:{v}, J:{J}, epsilon:{epsilon}')

    cmp2 = ['x', 'y', 'z']
    cmp1 = ['i', 'f']

    ds = xr.Dataset(
        data_vars={
            'r': (['cmp1', 'cmp2'], [ri, result['r']]),
            'p': (['cmp1', 'cmp2'], [pi, result['p']]),
            'R': (['cmp1', 'cmp2'], [Ri, result['R']]),
            'P': (['cmp1', 'cmp2'], [Pi, result['P']]),
            'tf':  result['tf'],
            'maxErr':  result['maxErr'],
            'countstep':  result['countstep'],
            'maxstep':  result['maxstep'],
            'converge':  result['converge'],
            'fail':  result['fail'],
            'l':  result['l'],
            'n':  result['n'],
            'case': result['case']
        },
        coords={
            'cmp2': cmp2,
            'cmp1': cmp1,
        }
    )
    return ds


def isConverged(dist):
    """Return True if trajectory is converged
    dist -- distance between particle and C.o.M of the other two
    """
    c1 = dist > c.Rcut
    return c1


def propagate(ri, pi, Ri, Pi, returnTraj=False):
    """Propagate trajectory from a given intial condition based on Hamilton's
    Equations.
    ri -- Initial C.o.M position for diatom (BC)
    pi -- Initial C.o.M conjugate momentum for diatom (BC)
    Ri -- Initial C.o.M position for scattering particle and diatom (BC)
    Pi -- Initial C.o.M conjugate momentum for scattering particle and diatom
          (BC)
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
    maxstep, maxErr = 0., 0.
    Rmax = 0.
    countstep = 0
    failed = False

    # propragate Hamilton's eqn's
    while stepper.t < c.tf:
        try:

            stepper.step()

            r, p, R, P = getPropCoords(stepper.y)
            R1, R2, R3 = internuclear(r, R)

            Rmax = np.max([R1, R2, R3])

            # force small step for close-encounters
#            if Rmax < 10.:
#                stepper.max_step = 1e-1
#            else:
#                stepper.max_step = c.maxstep

            trajectory.append([stepper.t] + stepper.y.tolist() + [H])

            maxstep = np.max([stepper.step_size, maxstep])
            maxErr = np.max(
                    [np.abs((H - num.Hamiltonian(r, p, R, P)/H)), maxErr]
                    )
            countstep = countstep + 1

            if isConverged(Rmax):
                break

            H = num.Hamiltonian(r, p, R, P)
        except RuntimeError as e:
            print(e)
            failed = True
            break

    if returnTraj:
        trajectory = np.array(trajectory)
        return trajectory
    else:
        assignment = assignClassical(r=r, p=p, R=R, P=P)
        nq, lq = assignQuantum(Ec=assignment["Ec"], lc=assignment["lc"])
        assignment["n"] = nq
        assignment["l"] = lq
        names = [
                    'r', 'p', 'R', 'P',
                    'tf', 'maxErr', 'countstep', 'maxstep',
                    'converge', 'fail',
                    "n", "l", "case",
                ]
        objects = [
                    r, p, R, P,
                    stepper.t, maxErr, countstep, maxstep,
                    isConverged(Rmax), failed,
                    assignment["n"], assignment["l"], assignment["case"],
                ]
        result = dict(zip(names, objects))
        return result
