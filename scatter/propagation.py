import numpy as np
from numpy.linalg import norm
from scipy.constants import codata
from scipy.special import legendre
import scipy.integrate as odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
# import xarray as xr


# TODO:
# - debug the potential
# - check numerical derivative
# - plot Vppp and dVppp in matplotlib
# - equations of motion need double checking

# NOTE:
# - currently code uses r, R and cos(gamma)
# - would need to change if gamma is needed
# - checked derivatives (numeric vs analytic)


# constants taken from NIST (in u)
# m_H2plus = 2.01533 u
# m_H = 1.00794 u
# m_p = 1.00728 u
# conversion from u to a.u.
m_e = codata.value('electron mass in u')
conversion = 1./m_e
MU_1 = 0.671606*conversion
MU_2 = 0.503805*conversion


def potential(r, R, gamma):
    alpha = 2.027
    beta = 0.375
    C = 17.283
    p2 = legendre(2)
    V = C*np.exp(-alpha*R)*(1.+beta*p2(cos(gamma)))
    return V


def derivative(func, x, h=1e-8):
    return 0.5*(func(x+h)-func(x-h))/h


def numericDerivatives(r, R):
    rmag = norm(r)
    Rmag = norm(R)
    cosGamma = (r @ R)/(rmag * Rmag)
    gamma = np.arccos(cosGamma)

    dVdr = derivative(lambda x: potential(x, Rmag, gamma), rmag)
    dVdR = derivative(lambda x: potential(rmag, x, gamma), Rmag)
    dVdgamma = derivative(lambda x: potential(rmag, Rmag, x), gamma)
    dgammadr = -(rmag/Rmag*R - cosGamma*r)/(rmag*rmag*np.sqrt(1-gamma*gamma))
    dgammadR = -(Rmag/rmag*r - cosGamma*R)/(Rmag*Rmag*np.sqrt(1-gamma*gamma))
    pdot = -dVdr*r/rmag - dVdgamma*dgammadr
    Pdot = -dVdR*R/Rmag - dVdgamma*dgammadR

    return pdot, Pdot

def analyticDerivatives(r, R):
    alpha = 2.027
    beta = 0.375
    C = 17.283

    rmag = norm(r)
    Rmag = norm(R)
    cosGamma = (r @ R)/(rmag * Rmag)

    Runit = R/Rmag

    dVdR = -alpha*potential(rmag, Rmag, cosGamma)
    dVdcosGamma = 3.*beta*C*np.exp(-alpha*Rmag)*cosGamma
    dcosGammadr = (rmag/Rmag*R - cosGamma*r)/(Rmag*Rmag)
    dcosGammadR = (Rmag/rmag*r - cosGamma*R)/(rmag*rmag)

    pdot = -dVdcosGamma*dcosGammadr
    Pdot = -dVdR*Runit -dVdcosGamma*dcosGammadR
    return pdot, Pdot


def equation_of_motion(t, coordinates, numeric=True):
    # current jacobi coordinates
    r, p, R, P = getJacobiCoordinates(coordinates)

    # first derivatives w.r.t. t
    rdot = p/MU_1
    Rdot = P/MU_2

    if numeric:
        pdot, Pdot = numericDerivatives(r, R)
        ana1, ana2 = analyticDerivatives(r, R)
#        print('pdot-analytic=', pdot-ana1)
#        print('Pdot-analytic=', Pdot-ana2)
    else:
        pdot, Pdot = analyticDerivatives(r, R)

    return np.concatenate((rdot, pdot, Rdot, Pdot), axis=0)


def getJacobiCoordinates(coordinates):
    r = coordinates[:3]
    p = coordinates[3:6]
    R = coordinates[6:9]
    P = coordinates[9:]

    return r, p, R, P


def getParticleCoordinates(r, p, R, P):
    r1 = -0.5*r
    r2 = 0.5*r
    r3 = R

    return r1, r2, r3


def testDeriv():
    N = 100
    r = np.linspace(0, 10, N)
    R = np.linspace(0.5, 10, N)
    cosGamma = np.linspace(-1, 1, N)
    grid_r, grid_R, grid_cosGamma = np.meshgrid(r, R, cosGamma)
    grid_Pot = potential(grid_r, grid_R, cosGamma)
    fig = plt.figure()
    stride = 1
    strideAna = 5

    alpha = 2.027
    beta = 0.375
    C = 17.283

    dVdr = derivative(lambda x: potential(x, grid_R, grid_cosGamma), grid_r)
    dVdrAnalytic = np.zeros(dVdr.shape)
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_wireframe(
            grid_r[:, :, 0],
            grid_R[:, :, 0],
            dVdr[:, :, 0],
            rstride=stride,
            cstride=stride
            )
    ax.plot_wireframe(
            grid_r[:, :, 0],
            grid_R[:, :, 0],
            dVdrAnalytic[:, :, 0],
            rstride=strideAna,
            cstride=strideAna,
            color='r'
            )
    dVdR = derivative(lambda x: potential(grid_r, x, grid_cosGamma), grid_R)
    dVdRAnalytic = -alpha*potential(grid_r, grid_R, grid_cosGamma)
    ax = fig.add_subplot(132, projection='3d')
    ax.plot_wireframe(
            grid_r[:, :, 0],
            grid_R[:, :, 0],
            dVdR[:, :, 0],
            rstride=stride,
            cstride=stride
            )
    ax.plot_wireframe(
            grid_r[:, :, 0],
            grid_R[:, :, 0],
            dVdRAnalytic[:, :, 0],
            rstride=strideAna,
            cstride=strideAna,
            color='r'
            )
    dVdcosGamma = derivative(lambda x: potential(grid_r, grid_R, x), grid_cosGamma)
    dVdcosGammaAnalytic = 3.*beta*C*np.exp(-alpha*grid_R)*grid_cosGamma
    ax = fig.add_subplot(133, projection='3d')
    ax.plot_wireframe(
            grid_cosGamma[:, 0, :],
            grid_R[:, 0, :],
            dVdcosGamma[:, 0, :],
            rstride=stride,
            cstride=stride
            )
    ax.plot_wireframe(
            grid_cosGamma[:, 0, :],
            grid_R[:, 0, :],
            dVdcosGammaAnalytic[:, 0, :],
            rstride=strideAna,
            cstride=strideAna,
            color='r'
            )
    plt.legend(['numeric','analytic'])
    plt.show()
    return


def main(args):
#    testDeriv()

    # time range
    ts = 0.
    tf = 100000.

    # initial conditions of the scattering particles
    r = np.array([2., 0., 0.], dtype=float)
    p = np.array([0., 0., 0.], dtype=float)
    R = np.array([0., 0., 3.], dtype=float)
    P = np.array([0., 0., -10.], dtype=float)
    initialConditions = np.concatenate((r, p, R, P), axis=0)

    # initialise stepping object
    stepper = odeint.RK45(
            lambda t, y: equation_of_motion(t, y, numeric=True),
            ts,
            initialConditions,
            tf,
            rtol=1e-08,
            atol=1e-10
            )

    # array to store trajectories
    data = []

    # propragate Hamilton's eqn's
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    while stepper.t < tf:
        try:
            stepper.step()
            R1, P1, R2, P2 = getJacobiCoordinates(stepper.y)
            r1, r2, r3 = getParticleCoordinates(R1, P1, R2, P2)
            ax.scatter(r1[0], r1[1], r1[2], c='r', marker='.')
            ax.scatter(r2[0], r2[1], r2[2], c='b', marker='.')
            ax.scatter(r3[0], r3[1], r3[2], c='k', marker='.')
            plt.pause(0.1)
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
