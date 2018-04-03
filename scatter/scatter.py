import numpy as np
from numpy import sin, cos
import scipy.integrate as odeint
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from scatter.analytics import equation_of_motion
from scatter.transformation import getPropCoords, getParticleCoords
import sys
# import xarray as xr


# TODO:

# NOTE:
# - when dr is set equal to 0. instead of pr/mu
#   the Hamiltonian is no longer conserved!!


m1 = 1.
m2 = 1.
mu = m1*m2/(m1+m2)
M = 1.


def derivative(func, x, h=1e-8):
    return 0.5*(func(x+h)-func(x-h))/h


def main(args):
    # run scattering calculation

    # time range
    ts = 0.
    tf = 1000.

    # initial conditions of the scattering particles
    # r, theta, R, phi, pr, ptheta, pR, pphi
    r_ini = np.array([2., 0., np.pi/2.])
    R_ini = np.array([5., 0., 0.])
    p_ini = np.array([0., 0.1, 0.])
    P_ini = np.array([-0.1, 0., 0.])

    r_ini = np.array([2., 0., np.pi/2.])
    R_ini = np.array([1./sin(np.pi/20.), 0., np.pi/20.])
    p_ini = np.array([0., 0., 0.])
    P_ini = M*np.array([-cos(R_ini[2]), 0., R_ini[0]*sin(R_ini[2])])

#    r_ini = np.array([2., 0., np.pi/2.])
#    R_ini = np.array([5., 0., 0.])
#    p_ini = np.array([0., 0.1, 0.1])
#    P_ini = np.array([-0.1, 0., 0.])

    initialConditions = np.hstack((r_ini, R_ini, p_ini, P_ini))

    # initialise stepping object
    stepper = odeint.RK45(
            lambda t, y: equation_of_motion(t, y, numeric=False),
            ts,
            initialConditions,
            tf,
            max_step=1e1,
            rtol=1e-06,
            atol=1e-08
            )

    # array to store trajectories
    data = []

    # propragate Hamilton's eqn's
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    while stepper.t < tf:
        try:
            stepper.step()
            print(stepper.step_size)
            r, t, p, R, T, P, pr, pt, pp, pR, pT, pP = getPropCoords(stepper.y)
            r1, r2, r3 = getParticleCoords(
                    r, t, p, R, T, P,
                    pr, pt, pp, pR, pT, pP
                    )
            ax.scatter(r1[0], r1[1], r1[2], c='r', marker='.')
            ax.scatter(r2[0], r2[1], r2[2], c='b', marker='.')
            ax.scatter(r3[0], r3[1], r3[2], c='k', marker='.')
            plt.pause(0.01)
            data.append([stepper.t] + stepper.y.tolist())
        except RuntimeError as e:
            print(e)
            break
    plt.ioff()
    data = np.array(data)

#    legend = []
#    for i in range(1, 4):
#        plt.plot(data[:, 0], data[:, i], 'k')
#        plt.plot(dataAnalytic[:, 0], dataAnalytic[:, i], '--r')
#        legend.append(f'r1{i}_n')
#        legend.append(f'r1{i}_a')
#    plt.legend(legend)
#    plt.show()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
