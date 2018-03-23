import numpy as np
from numpy.linalg import norm
import scipy.integrate as odeint
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import sys
# import xarray as xr


def potentialAnalytic(r1, k=1.):
    return 0.5*k*r1 @ r1


def potentialNumeric(r1, k=1.):
    return 0.5*k*r1*r1


def dVdrGridNumeric(r1, k=1.):
    return np.gradient(potentialNumeric(r1), r1)


def dVdrGridAnalytic(r1, k=1.):
    return k*r1


def dVdrAnalytic(r1, k=1.):
    return k*r1


def dVdrNumeric(r1, interp, k=1.):
    if norm(r1) > 0:
        unit = r1/norm(r1)
    else:
        unit = r1
    return k*interp(norm(r1))*unit


def equation_of_motion(t, coordinates, interp, analytic=False):
    m1 = 1.
    r1 = coordinates[:3]
    p1 = coordinates[3:]
    r1dot = p1/m1
    if analytic:
        p1dot = -dVdrAnalytic(r1)
    else:
        p1dot = -dVdrNumeric(r1, interp)
    return np.hstack((r1dot, p1dot))


def main(args):
    # initial conditions of the scattering particle
    ts = 0.
    tf = 10.
    r1 = np.array([0., 0., 0.], dtype=float)
    p1 = np.array([1., 0.5, 0.], dtype=float)
    initialConditions = np.hstack((r1, p1))

    # initialise interpolater for dVdr
    # for now a 1D grid is assumed (but this can be extended)
    gridPoints = np.linspace(-5., 5., 50)
    interp = interpolate.interp1d(
            gridPoints,
            dVdrGridNumeric(gridPoints),
            kind='cubic'
            )

    # initialise stepping object
    stepper = odeint.RK45(
            lambda t, y: equation_of_motion(t, y, interp, analytic=False),
            ts,
            initialConditions,
            tf,
            rtol=1e-08,
            atol=1e-10
            )

    # array to store trajectories
    data = []

    # propragate Hamilton's eqn's
    while stepper.t < tf:
        try:
            stepper.step()
            data.append([stepper.t] + stepper.y.tolist())
        except RuntimeError as e:
            print(e)
            break
    data = np.array(data)

    # initialise stepping object
    stepper = odeint.RK45(
            lambda t, y: equation_of_motion(t, y, interp, analytic=True),
            ts,
            initialConditions,
            tf,
            rtol=1e-08,
            atol=1e-10
            )

    # array to store trajectories
    dataAnalytic = []

    # propragate Hamilton's eqn's
    while stepper.t < tf:
        try:
            stepper.step()
            dataAnalytic.append([stepper.t] + stepper.y.tolist())
        except RuntimeError as e:
            print(e)
            break

    dataAnalytic = np.array(dataAnalytic)
    legend = []
    for i in range(1, 4):
        plt.plot(data[:, 0], data[:, i], 'k')
        plt.plot(dataAnalytic[:, 0], dataAnalytic[:, i], '--r')
        legend.append(f'r1{i}_n')
        legend.append(f'r1{i}_a')
    plt.legend(legend)
    plt.show()
#    plt.plot(gridPoints, potentialGrid(gridPoints), '-x')
#    plt.plot(gridPoints, dVdrGridAnalytic(gridPoints), '-x')
#    plt.plot(gridPoints, dVdrGrid(gridPoints), '-s')
#    plt.legend(['V(r)', 'ana', 'num'])
#    plt.show()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
